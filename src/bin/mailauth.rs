/*
 * CLI wrapper around the mail-auth library for PortBench.
 *
 * Provides deterministic DKIM signing/verification and ARC sealing/verification
 * using local key files instead of DNS lookups.
 *
 * SPDX-License-Identifier: Apache-2.0 OR MIT
 */

use std::{
    borrow::Borrow,
    collections::HashMap,
    hash::Hash,
    io::{self, Read},
    net::{IpAddr, Ipv4Addr, Ipv6Addr},
    process,
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};

use clap::{Parser, Subcommand, ValueEnum};
use mail_auth::{
    AuthenticatedMessage, AuthenticationResults, DkimResult, MX, MessageAuthenticator, Parameters,
    ResolverCache, Txt,
    arc::ArcSealer,
    common::{
        crypto::{RsaKey, Sha256},
        headers::{HeaderIterator, HeaderWriter, Writable},
        parse::TxtRecordParser,
        resolver::ToFqdn,
        verify::{DomainKey, VerifySignature},
    },
    dkim::{Canonicalization, DkimSigner},
};
use rustls_pki_types::{PrivateKeyDer, PrivatePkcs1KeyDer, pem::PemObject};

// ---------------------------------------------------------------------------
// CLI argument parsing
// ---------------------------------------------------------------------------

#[derive(Parser)]
#[command(name = "mailauth", about = "DKIM/ARC signing and verification tool")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Sign a message with DKIM. Reads message from stdin, writes signed message to stdout.
    DkimSign {
        /// Path to PEM-encoded RSA private key file
        #[arg(long)]
        key: String,

        /// Signing domain (d= tag)
        #[arg(long)]
        domain: String,

        /// Selector (s= tag)
        #[arg(long)]
        selector: String,

        /// Headers to sign (comma-separated, e.g. "From,To,Subject")
        #[arg(long, value_delimiter = ',')]
        headers: Vec<String>,

        /// Unix timestamp for the signature (t= tag). Required for deterministic output.
        #[arg(long)]
        timestamp: u64,

        /// Header canonicalization algorithm
        #[arg(long, default_value = "relaxed")]
        header_canon: Canon,

        /// Body canonicalization algorithm
        #[arg(long, default_value = "relaxed")]
        body_canon: Canon,
    },

    /// Verify DKIM signatures. Reads message from stdin.
    /// Prints one line per signature: "pass <domain>" or "fail <domain>: <reason>".
    /// Exit code 0 if all signatures pass, 1 otherwise.
    DkimVerify {
        /// Path to DNS records file (one record per line: "selector._domainkey.domain <TXT value>")
        #[arg(long)]
        dns_file: String,
    },

    /// Seal a message with ARC. Reads message from stdin, writes sealed message to stdout.
    /// Verifies existing DKIM/ARC chain, adds a new ARC set.
    ArcSeal {
        /// Path to PEM-encoded RSA private key file
        #[arg(long)]
        key: String,

        /// Signing domain (d= tag)
        #[arg(long)]
        domain: String,

        /// Selector (s= tag)
        #[arg(long)]
        selector: String,

        /// Headers to sign in ARC-Message-Signature (comma-separated)
        #[arg(long, value_delimiter = ',')]
        headers: Vec<String>,

        /// Unix timestamp. Required for deterministic output.
        #[arg(long)]
        timestamp: u64,

        /// Authentication service identifier for ARC-Authentication-Results
        #[arg(long)]
        srv_id: String,

        /// Path to DNS records file
        #[arg(long)]
        dns_file: String,

        /// Header canonicalization algorithm
        #[arg(long, default_value = "relaxed")]
        header_canon: Canon,

        /// Body canonicalization algorithm
        #[arg(long, default_value = "relaxed")]
        body_canon: Canon,
    },

    /// Verify an ARC chain. Reads message from stdin.
    /// Prints the chain validation result. Exit code 0 if pass or none, 1 otherwise.
    ArcVerify {
        /// Path to DNS records file
        #[arg(long)]
        dns_file: String,
    },

    /// Canonicalize a message. Reads message from stdin, writes canonicalized message to stdout.
    /// Useful for debugging: compare canonicalized output to see where implementations diverge.
    Canonicalize {
        /// Header canonicalization algorithm
        #[arg(long, default_value = "relaxed")]
        header_canon: Canon,

        /// Body canonicalization algorithm
        #[arg(long, default_value = "relaxed")]
        body_canon: Canon,
    },
}

#[derive(Clone, ValueEnum)]
enum Canon {
    Relaxed,
    Simple,
}

impl From<&Canon> for Canonicalization {
    fn from(c: &Canon) -> Self {
        match c {
            Canon::Relaxed => Canonicalization::Relaxed,
            Canon::Simple => Canonicalization::Simple,
        }
    }
}

// ---------------------------------------------------------------------------
// Local DNS cache (replaces network DNS lookups with file-based records)
// ---------------------------------------------------------------------------

struct LocalCache<K, V>(Mutex<HashMap<K, V>>);

impl<K: Hash + Eq, V: Clone> LocalCache<K, V> {
    fn new() -> Self {
        LocalCache(Mutex::new(HashMap::new()))
    }
}

impl<K: Hash + Eq, V: Clone> ResolverCache<K, V> for LocalCache<K, V> {
    fn get<Q>(&self, key: &Q) -> Option<V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        self.0.lock().unwrap().get(key).cloned()
    }

    fn remove<Q>(&self, key: &Q) -> Option<V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        self.0.lock().unwrap().remove(key)
    }

    fn insert(&self, key: K, value: V, _valid_until: Instant) {
        self.0.lock().unwrap().insert(key, value);
    }
}

struct LocalCaches {
    txt: LocalCache<Box<str>, Txt>,
    mx: LocalCache<Box<str>, Arc<[MX]>>,
    ptr: LocalCache<IpAddr, Arc<[Box<str>]>>,
    ipv4: LocalCache<Box<str>, Arc<[Ipv4Addr]>>,
    ipv6: LocalCache<Box<str>, Arc<[Ipv6Addr]>>,
}

impl LocalCaches {
    fn new() -> Self {
        Self {
            txt: LocalCache::new(),
            mx: LocalCache::new(),
            ptr: LocalCache::new(),
            ipv4: LocalCache::new(),
            ipv6: LocalCache::new(),
        }
    }

    fn parameters<T>(
        &self,
        param: T,
    ) -> Parameters<
        '_,
        T,
        LocalCache<Box<str>, Txt>,
        LocalCache<Box<str>, Arc<[MX]>>,
        LocalCache<Box<str>, Arc<[Ipv4Addr]>>,
        LocalCache<Box<str>, Arc<[Ipv6Addr]>>,
        LocalCache<IpAddr, Arc<[Box<str>]>>,
    > {
        Parameters::new(param)
            .with_txt_cache(&self.txt)
            .with_mx_cache(&self.mx)
            .with_ptr_cache(&self.ptr)
            .with_ipv4_cache(&self.ipv4)
            .with_ipv6_cache(&self.ipv6)
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn read_stdin() -> Vec<u8> {
    let mut buf = Vec::new();
    io::stdin().read_to_end(&mut buf).unwrap_or_else(|e| {
        eprintln!("Error reading stdin: {e}");
        process::exit(2);
    });
    buf
}

/// Normalize LF line endings to CRLF (as required by RFC 5322 / DKIM / ARC).
fn normalize_crlf(input: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(input.len() + input.len() / 20);
    let mut prev = 0u8;
    for &b in input {
        if b == b'\n' && prev != b'\r' {
            out.push(b'\r');
        }
        out.push(b);
        prev = b;
    }
    out
}

fn load_rsa_key(path: &str) -> RsaKey<Sha256> {
    let pem = std::fs::read_to_string(path).unwrap_or_else(|e| {
        eprintln!("Error reading key file '{path}': {e}");
        process::exit(2);
    });
    RsaKey::<Sha256>::from_key_der(PrivateKeyDer::Pkcs1(
        PrivatePkcs1KeyDer::from_pem_slice(pem.as_bytes()).unwrap_or_else(|e| {
            eprintln!("Error parsing PEM key '{path}': {e}");
            process::exit(2);
        }),
    ))
    .unwrap_or_else(|e| {
        eprintln!("Error loading RSA key '{path}': {e}");
        process::exit(2);
    })
}

/// Load DNS TXT records from a file.
/// Format: one record per line, `<name> <TXT record value>`
/// Lines starting with '#' or empty lines are skipped.
fn load_dns_caches(path: &str) -> LocalCaches {
    let contents = std::fs::read_to_string(path).unwrap_or_else(|e| {
        eprintln!("Error reading DNS file '{path}': {e}");
        process::exit(2);
    });
    let caches = LocalCaches::new();
    let far_future = Instant::now() + Duration::from_secs(86400);

    for line in contents.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let (name, value) = line.split_once(' ').unwrap_or_else(|| {
            eprintln!("Invalid DNS record line (expected '<name> <value>'): {line}");
            process::exit(2);
        });
        let dk = DomainKey::parse(value.as_bytes()).unwrap_or_else(|e| {
            eprintln!("Error parsing DKIM key for '{name}': {e}");
            process::exit(2);
        });
        // Ensure FQDN (trailing dot)
        let fqdn = name.to_fqdn();
        caches.txt.insert(fqdn, dk.into(), far_future);
    }
    caches
}

// ---------------------------------------------------------------------------
// Subcommand implementations
// ---------------------------------------------------------------------------

fn cmd_dkim_sign(
    key_path: &str,
    domain: &str,
    selector: &str,
    headers: &[String],
    timestamp: u64,
    header_canon: &Canon,
    body_canon: &Canon,
) {
    let pk = load_rsa_key(key_path);
    let raw = read_stdin();
    let message = normalize_crlf(&raw);

    let signature = DkimSigner::from_key(pk)
        .domain(domain)
        .selector(selector)
        .headers(headers.iter().map(|s| s.as_str()))
        .header_canonicalization(header_canon.into())
        .body_canonicalization(body_canon.into())
        .sign_stream(
            HeaderIterator::new(&message),
            timestamp,
        )
        .unwrap_or_else(|e| {
            eprintln!("Error signing message: {e}");
            process::exit(1);
        });

    // Output: DKIM-Signature header + original message
    print!("{}", signature.to_header());
    // Write the original message bytes (already CRLF-normalized)
    io::Write::write_all(&mut io::stdout(), &message).unwrap();
}

async fn cmd_dkim_verify(dns_file: &str) {
    let caches = load_dns_caches(dns_file);
    let raw = read_stdin();
    let message = normalize_crlf(&raw);

    let authenticated = AuthenticatedMessage::parse(&message).unwrap_or_else(|| {
        eprintln!("Error: could not parse message");
        process::exit(2);
    });

    let resolver = MessageAuthenticator::new_system_conf().unwrap_or_else(|e| {
        eprintln!("Error creating resolver: {e}");
        process::exit(2);
    });

    let results = resolver
        .verify_dkim(caches.parameters(&authenticated))
        .await;

    let mut all_pass = true;
    if results.is_empty() {
        println!("none");
        all_pass = false;
    }
    for output in &results {
        let domain = output
            .signature()
            .map(|s| s.domain().to_string())
            .unwrap_or_else(|| "unknown".to_string());
        match output.result() {
            DkimResult::Pass => println!("pass {domain}"),
            other => {
                println!("fail {domain}: {other}");
                all_pass = false;
            }
        }
    }

    if !all_pass {
        process::exit(1);
    }
}

async fn cmd_arc_seal(
    key_path: &str,
    domain: &str,
    selector: &str,
    headers: &[String],
    timestamp: u64,
    srv_id: &str,
    dns_file: &str,
    header_canon: &Canon,
    body_canon: &Canon,
) {
    let pk = load_rsa_key(key_path);
    let caches = load_dns_caches(dns_file);
    let raw = read_stdin();
    let message_bytes = normalize_crlf(&raw);

    let authenticated = AuthenticatedMessage::parse(&message_bytes).unwrap_or_else(|| {
        eprintln!("Error: could not parse message");
        process::exit(2);
    });

    let resolver = MessageAuthenticator::new_system_conf().unwrap_or_else(|e| {
        eprintln!("Error creating resolver: {e}");
        process::exit(2);
    });

    // Verify existing DKIM signatures and ARC chain
    let dkim_result = resolver
        .verify_dkim(caches.parameters(&authenticated))
        .await;
    let arc_result = resolver
        .verify_arc(caches.parameters(&authenticated))
        .await;

    // Build Authentication-Results
    let auth_results = AuthenticationResults::new(srv_id)
        .with_dkim_results(&dkim_result, srv_id);

    if !arc_result.can_be_sealed() {
        eprintln!(
            "Error: ARC chain cannot be sealed (cv=fail). ARC result: {}",
            arc_result.result()
        );
        process::exit(1);
    }

    let arc_set = ArcSealer::from_key(pk)
        .domain(domain)
        .selector(selector)
        .headers(headers.iter().map(|s| s.as_str()))
        .header_canonicalization(header_canon.into())
        .body_canonicalization(body_canon.into())
        .seal_at(&authenticated, &auth_results, &arc_result, timestamp)
        .unwrap_or_else(|e| {
            eprintln!("Error sealing message: {e}");
            process::exit(1);
        });

    // Output: ARC headers + Authentication-Results + original message
    print!("{}", arc_set.to_header());
    print!("{}", auth_results.to_header());
    io::Write::write_all(&mut io::stdout(), &message_bytes).unwrap();
}

async fn cmd_arc_verify(dns_file: &str) {
    let caches = load_dns_caches(dns_file);
    let raw = read_stdin();
    let message = normalize_crlf(&raw);

    let authenticated = AuthenticatedMessage::parse(&message).unwrap_or_else(|| {
        eprintln!("Error: could not parse message");
        process::exit(2);
    });

    let resolver = MessageAuthenticator::new_system_conf().unwrap_or_else(|e| {
        eprintln!("Error creating resolver: {e}");
        process::exit(2);
    });

    let result = resolver
        .verify_arc(caches.parameters(&authenticated))
        .await;

    let num_sets = result.sets().len();
    match result.result() {
        DkimResult::Pass => {
            println!("pass ({num_sets} sets)");
        }
        DkimResult::None => {
            println!("none");
        }
        other => {
            println!("fail: {other}");
            process::exit(1);
        }
    }
}

fn cmd_canonicalize(header_canon: &Canon, body_canon: &Canon) {
    let raw = read_stdin();
    let message = normalize_crlf(&raw);

    let ch: Canonicalization = header_canon.into();
    let cb: Canonicalization = body_canon.into();

    let mut header_iter = HeaderIterator::new(&message);
    let headers: Vec<_> = (&mut header_iter).collect();
    let body = header_iter
        .body_offset()
        .map(|pos| &message[pos..])
        .unwrap_or_default();

    let mut output: Vec<u8> = Vec::new();
    ch.canonicalize_headers(headers.into_iter(), &mut output);
    output.extend_from_slice(b"\r\n");
    cb.canonical_body(body, 0).write(&mut output);

    io::Write::write_all(&mut io::stdout(), &output).unwrap();
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

#[tokio::main]
async fn main() {
    let cli = Cli::parse();

    match &cli.command {
        Commands::DkimSign {
            key,
            domain,
            selector,
            headers,
            timestamp,
            header_canon,
            body_canon,
        } => cmd_dkim_sign(key, domain, selector, headers, *timestamp, header_canon, body_canon),

        Commands::DkimVerify { dns_file } => cmd_dkim_verify(dns_file).await,

        Commands::ArcSeal {
            key,
            domain,
            selector,
            headers,
            timestamp,
            srv_id,
            dns_file,
            header_canon,
            body_canon,
        } => {
            cmd_arc_seal(
                key,
                domain,
                selector,
                headers,
                *timestamp,
                srv_id,
                dns_file,
                header_canon,
                body_canon,
            )
            .await
        }

        Commands::ArcVerify { dns_file } => cmd_arc_verify(dns_file).await,

        Commands::Canonicalize {
            header_canon,
            body_canon,
        } => cmd_canonicalize(header_canon, body_canon),
    }
}

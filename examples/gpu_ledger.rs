//! GPU VRAM Ledger Example (GPU-SHARE-001)
//!
//! Demonstrates the VRAM reservation ledger for GPU sharing.
//!
//! ```bash
//! cargo run --example gpu_ledger
//! cargo run --example gpu_ledger -- --reserve 8000 --task "qlora-7b"
//! cargo run --example gpu_ledger -- --status
//! cargo run --example gpu_ledger -- --release
//! ```

fn main() {
    let args: Vec<String> = std::env::args().collect();

    if args.iter().any(|a| a == "--help" || a == "-h") {
        println!("GPU VRAM Ledger Example");
        println!();
        println!("Usage:");
        println!("  gpu_ledger                          # Auto-detect GPU, show status");
        println!("  gpu_ledger --status                 # Show ledger status");
        println!("  gpu_ledger --reserve <MB> --task <name>  # Reserve VRAM");
        println!("  gpu_ledger --release                # Release our reservation");
        println!("  gpu_ledger --wait <MB>              # Wait for VRAM");
        return;
    }

    // Auto-detect GPU
    let uuid = entrenar::gpu::ledger::detect_gpu_uuid();
    let total_mb = entrenar::gpu::ledger::detect_total_memory_mb();
    let mem_type = entrenar::gpu::ledger::detect_memory_type();

    println!("GPU: {uuid}");
    println!("Total: {total_mb} MB");
    println!("Type: {mem_type:?} (reserve factor: {:.0}%)", mem_type.reserve_factor() * 100.0);
    println!();

    let mut ledger =
        entrenar::gpu::ledger::VramLedger::new(uuid, total_mb, mem_type.reserve_factor())
            .with_profiling(true);

    // --status: show current reservations
    if args.iter().any(|a| a == "--status") {
        match entrenar::gpu::ledger::gpu_status_display(&ledger) {
            Ok(status) => print!("{status}"),
            Err(e) => eprintln!("Error: {e}"),
        }
        return;
    }

    // --reserve <MB> --task <name>: create a reservation
    if let Some(pos) = args.iter().position(|a| a == "--reserve") {
        let budget_mb: usize = args
            .get(pos + 1)
            .and_then(|s| s.parse().ok())
            .unwrap_or_else(|| {
                eprintln!("--reserve requires a number (MB)");
                std::process::exit(1);
            });

        let task = args
            .iter()
            .position(|a| a == "--task")
            .and_then(|p| args.get(p + 1))
            .map(String::as_str)
            .unwrap_or("example");

        match ledger.try_reserve(budget_mb, task) {
            Ok(id) => {
                println!("Reserved {budget_mb} MB (id: {id})");
                println!("Press Enter to release...");
                let mut input = String::new();
                std::io::stdin().read_line(&mut input).ok();
                ledger.release().ok();
                println!("Released.");
            }
            Err(e) => eprintln!("Error: {e}"),
        }

        println!("\n{}", ledger.profiler_report());
        return;
    }

    // --wait <MB>: wait for VRAM availability
    if let Some(pos) = args.iter().position(|a| a == "--wait") {
        let budget_mb: usize = args
            .get(pos + 1)
            .and_then(|s| s.parse().ok())
            .unwrap_or_else(|| {
                eprintln!("--wait requires a number (MB)");
                std::process::exit(1);
            });

        let config = entrenar::gpu::wait::WaitConfig::with_timeout_secs(60);
        let mut profiler = entrenar::gpu::profiler::GpuProfiler::new(true);

        match entrenar::gpu::wait::wait_for_vram(
            &mut ledger,
            budget_mb,
            "wait-example",
            &config,
            &mut profiler,
        ) {
            Ok(id) => {
                println!("Reserved {budget_mb} MB after waiting (id: {id})");
                ledger.release().ok();
            }
            Err(e) => eprintln!("Error: {e}"),
        }
        return;
    }

    // Default: show status
    match entrenar::gpu::ledger::gpu_status_display(&ledger) {
        Ok(status) => print!("{status}"),
        Err(e) => eprintln!("Error: {e}"),
    }
}

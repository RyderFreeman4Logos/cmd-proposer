use anyhow::Result;
use clap::{Parser, Subcommand};

#[derive(Parser, Debug)]
#[command(
    name = "cps",
    version,
    about = "cmd-proposer — sandboxed argv proposer for production ops"
)]
struct Cli {
    #[command(subcommand)]
    cmd: Option<Command>,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Print a config template to stdout. Pipe into ./.cmd-proposer.yaml to bootstrap.
    Init {
        /// Print the full reference (all settings shown with defaults commented out).
        #[arg(long)]
        full: bool,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.cmd {
        Some(Command::Init { full }) => {
            let template = if full {
                cps_config::full_template()
            } else {
                cps_config::minimal_template()
            };
            print!("{template}");
        }
        None => {
            println!("cps {}", env!("CARGO_PKG_VERSION"));
            println!("(no subcommand — try `cps init` to scaffold a config)");
        }
    }
    Ok(())
}

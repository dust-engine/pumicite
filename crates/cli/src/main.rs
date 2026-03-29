use std::path::PathBuf;

use clap::{Parser, Subcommand, ValueEnum};
use pumicite_cli::build_pipeline_layout;
use shader_slang as slang;

#[derive(Parser)]
#[command(name = "pumicite", about = "Pumicite pipeline tooling")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Extract a PipelineLayout from a Slang shader module
    Slang(SlangArgs),
    /// Convert a .ron file containing pumicite types to postcard binary
    #[command(name = "ron2bin")]
    Ron2Bin(Ron2BinArgs),
}

// ---------------------------------------------------------------------------
// slang subcommand
// ---------------------------------------------------------------------------

#[derive(clap::Args)]
struct SlangArgs {
    /// Path to the .slang shader file
    shader: PathBuf,

    /// Additional search paths for #include / import resolution
    #[arg(short = 'I', long = "include")]
    include_paths: Vec<PathBuf>,

    /// Output file (defaults to stdout)
    #[arg(short, long)]
    output: Option<PathBuf>,

    /// Slang profile to compile against
    #[arg(short, long, default_value = "glsl_450")]
    profile: String,

    /// Optimization level
    #[arg(short = 'O', long, default_value = "high")]
    optimization: OptLevel,

    /// Output format
    #[arg(short, long, default_value = "ron")]
    format: OutputFormat,
}

#[derive(Clone, ValueEnum)]
enum OptLevel {
    None,
    Default,
    High,
    Maximal,
}

impl OptLevel {
    fn to_slang(&self) -> slang::OptimizationLevel {
        match self {
            OptLevel::None => slang::OptimizationLevel::None,
            OptLevel::Default => slang::OptimizationLevel::Default,
            OptLevel::High => slang::OptimizationLevel::High,
            OptLevel::Maximal => slang::OptimizationLevel::Maximal,
        }
    }
}

#[derive(Clone, ValueEnum)]
enum OutputFormat {
    Ron,
    Bin,
}

// ---------------------------------------------------------------------------
// ron2bin subcommand
// ---------------------------------------------------------------------------

#[derive(clap::Args)]
struct Ron2BinArgs {
    /// Input .ron file
    input: PathBuf,

    /// Output file (defaults to replacing the .ron extension with .bin)
    #[arg(short, long)]
    output: Option<PathBuf>,

    /// Type contained in the .ron file (inferred from file extension if omitted)
    #[arg(short, long)]
    r#type: Option<RonType>,
}

#[derive(Clone, ValueEnum)]
enum RonType {
    PipelineLayout,
    DescriptorSetLayout,
    ComputePipeline,
    GraphicsPipeline,
    RaytracingPipeline,
}

// ---------------------------------------------------------------------------
// write helpers
// ---------------------------------------------------------------------------

fn write_output(bytes: &[u8], path: Option<&PathBuf>, is_text: bool) {
    match path {
        Some(path) => {
            std::fs::write(path, bytes).unwrap_or_else(|e| {
                eprintln!("error: failed to write {path:?}: {e}");
                std::process::exit(1);
            });
        }
        None => {
            use std::io::Write;
            std::io::stdout().write_all(bytes).unwrap();
            if is_text {
                println!();
            }
        }
    }
}

// ---------------------------------------------------------------------------
// subcommand: slang
// ---------------------------------------------------------------------------

fn cmd_slang(args: SlangArgs) {
    let shader_path = args.shader.canonicalize().unwrap_or_else(|e| {
        eprintln!("error: cannot resolve shader path {:?}: {e}", args.shader);
        std::process::exit(1);
    });

    let global_session = slang::GlobalSession::new().unwrap();

    let mut search_path_cstrings = Vec::new();
    if let Some(parent) = shader_path.parent() {
        search_path_cstrings
            .push(std::ffi::CString::new(parent.to_string_lossy().as_ref()).unwrap());
    }
    for inc in &args.include_paths {
        search_path_cstrings.push(std::ffi::CString::new(inc.to_string_lossy().as_ref()).unwrap());
    }
    let search_path_ptrs: Vec<_> = search_path_cstrings.iter().map(|c| c.as_ptr()).collect();

    let session_options = slang::CompilerOptions::default()
        .optimization(args.optimization.to_slang())
        .matrix_layout_row(true);

    let target_desc = slang::TargetDesc::default()
        .format(slang::CompileTarget::Spirv)
        .profile(global_session.find_profile(&args.profile));

    let targets = [target_desc];

    let session_desc = slang::SessionDesc::default()
        .targets(&targets)
        .search_paths(&search_path_ptrs)
        .options(&session_options);

    let session = global_session.create_session(&session_desc).unwrap();

    let module_name = shader_path.to_string_lossy();
    let module = session.load_module(&module_name).unwrap_or_else(|e| {
        eprintln!("error: failed to load module {shader_path:?}: {e}");
        std::process::exit(1);
    });

    let entry_points: Vec<slang::EntryPoint> = module.entry_points().collect();
    if entry_points.is_empty() {
        eprintln!("warning: no entry points found in {shader_path:?}");
    }

    let mut components: Vec<slang::ComponentType> = vec![module.into()];
    for ep in entry_points {
        components.push(ep.into());
    }

    let program = session
        .create_composite_component_type(&components)
        .unwrap();
    let linked = program.link().unwrap();
    let reflection = linked.layout(0).unwrap();
    let layout = build_pipeline_layout(reflection);

    let (bytes, is_text) = match args.format {
        OutputFormat::Ron => (
            ron::ser::to_string_pretty(&layout, Default::default())
                .unwrap()
                .into_bytes(),
            true,
        ),
        OutputFormat::Bin => (postcard::to_allocvec(&layout).unwrap(), false),
    };

    write_output(&bytes, args.output.as_ref(), is_text);
}

// ---------------------------------------------------------------------------
// subcommand: ron2bin
// ---------------------------------------------------------------------------

fn infer_ron_type(path: &PathBuf) -> Option<RonType> {
    let name = path.to_string_lossy();
    if name.ends_with(".playout.ron") {
        Some(RonType::PipelineLayout)
    } else if name.ends_with(".desc.ron") {
        Some(RonType::DescriptorSetLayout)
    } else if name.ends_with(".comp.pipeline.ron") {
        Some(RonType::ComputePipeline)
    } else if name.ends_with(".gfx.pipeline.ron") {
        Some(RonType::GraphicsPipeline)
    } else if name.ends_with(".rtx.pipeline.ron") {
        Some(RonType::RaytracingPipeline)
    } else {
        None
    }
}

fn default_postcard_output(input: &PathBuf) -> PathBuf {
    let name = input.to_string_lossy();
    if let Some(stripped) = name.strip_suffix(".ron") {
        PathBuf::from(format!("{stripped}.bin"))
    } else {
        input.with_extension("bin")
    }
}

macro_rules! ron_to_postcard {
    ($ron:expr, $ty:ty) => {{
        let val: $ty = ron::de::from_str(&$ron).unwrap_or_else(|e| {
            eprintln!("error: failed to parse RON as {}: {e}", stringify!($ty));
            std::process::exit(1);
        });
        postcard::to_allocvec(&val).unwrap()
    }};
}

fn cmd_ron2bin(args: Ron2BinArgs) {
    let ron_type = args
        .r#type
        .or_else(|| infer_ron_type(&args.input))
        .unwrap_or_else(|| {
            eprintln!(
                "error: cannot infer type from file name {:?}. \
             Use --type to specify one of: pipeline-layout, descriptor-set-layout, \
             compute-pipeline, graphics-pipeline, ray-tracing-pipeline",
                args.input
            );
            std::process::exit(1);
        });

    let ron_str = std::fs::read_to_string(&args.input).unwrap_or_else(|e| {
        eprintln!("error: failed to read {:?}: {e}", args.input);
        std::process::exit(1);
    });

    let bytes = match ron_type {
        RonType::PipelineLayout => ron_to_postcard!(ron_str, pumicite_types::PipelineLayout),
        RonType::DescriptorSetLayout => {
            ron_to_postcard!(ron_str, pumicite_types::DescriptorSetLayout)
        }
        RonType::ComputePipeline => ron_to_postcard!(ron_str, pumicite_types::ComputePipeline),
        RonType::GraphicsPipeline => ron_to_postcard!(ron_str, pumicite_types::GraphicsPipeline),
        RonType::RaytracingPipeline => {
            ron_to_postcard!(ron_str, pumicite_types::RayTracingPipeline)
        }
    };

    let output = args
        .output
        .unwrap_or_else(|| default_postcard_output(&args.input));
    write_output(&bytes, Some(&output), false);
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

fn main() {
    let cli = Cli::parse();
    match cli.command {
        Command::Slang(args) => cmd_slang(args),
        Command::Ron2Bin(args) => cmd_ron2bin(args),
    }
}

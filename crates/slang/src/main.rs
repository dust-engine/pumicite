use std::path::PathBuf;

use clap::Parser;
use shader_slang as slang;

#[derive(Parser)]
#[command(
    name = "pumicite-slang",
    about = "Extract a PipelineLayout from a Slang shader module"
)]
struct Cli {
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

    /// Optimization level (none, default, high, maximal)
    #[arg(short = 'O', long, default_value = "high")]
    optimization: OptLevel,

    /// Output format (ron, postcard)
    #[arg(short, long, default_value = "ron")]
    format: OutputFormat,
}

#[derive(Clone, clap::ValueEnum)]
enum OutputFormat {
    Ron,
    Postcard,
}

#[derive(Clone, clap::ValueEnum)]
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

fn main() {
    let cli = Cli::parse();

    let shader_path = cli.shader.canonicalize().unwrap_or_else(|e| {
        eprintln!("error: cannot resolve shader path {:?}: {e}", cli.shader);
        std::process::exit(1);
    });

    let global_session = slang::GlobalSession::new().unwrap();

    // Build search paths: the shader's parent directory + any user-supplied -I paths.
    let mut search_path_cstrings = Vec::new();
    if let Some(parent) = shader_path.parent() {
        search_path_cstrings
            .push(std::ffi::CString::new(parent.to_string_lossy().as_ref()).unwrap());
    }
    for inc in &cli.include_paths {
        search_path_cstrings.push(std::ffi::CString::new(inc.to_string_lossy().as_ref()).unwrap());
    }
    let search_path_ptrs: Vec<_> = search_path_cstrings.iter().map(|c| c.as_ptr()).collect();

    let session_options = slang::CompilerOptions::default()
        .optimization(cli.optimization.to_slang())
        .matrix_layout_row(true);

    let target_desc = slang::TargetDesc::default()
        .format(slang::CompileTarget::Spirv)
        .profile(global_session.find_profile(&cli.profile));

    let targets = [target_desc];

    let session_desc = slang::SessionDesc::default()
        .targets(&targets)
        .search_paths(&search_path_ptrs)
        .options(&session_options);

    let session = global_session.create_session(&session_desc).unwrap();

    let module_name = shader_path.to_string_lossy();
    let module = session.load_module(&module_name).unwrap_or_else(|e| {
        eprintln!("error: failed to load module {:?}: {e}", shader_path);
        std::process::exit(1);
    });

    // Collect all entry points from the module.
    let entry_points: Vec<slang::EntryPoint> = module.entry_points().collect();
    if entry_points.is_empty() {
        eprintln!("warning: no entry points found in {:?}", shader_path);
    }

    // Build a composite component type from the module and its entry points.
    let mut components: Vec<slang::ComponentType> = vec![module.into()];
    for ep in entry_points {
        components.push(ep.into());
    }

    let program = session
        .create_composite_component_type(&components)
        .unwrap();
    let linked = program.link().unwrap();

    let reflection = linked.layout(0).unwrap();
    let layout = pumicite_slang::build_pipeline_layout(reflection);

    let output_bytes: Vec<u8> = match cli.format {
        OutputFormat::Ron => ron::ser::to_string_pretty(&layout, Default::default())
            .unwrap()
            .into_bytes(),
        OutputFormat::Postcard => postcard::to_allocvec(&layout).unwrap(),
    };

    match cli.output {
        Some(path) => {
            std::fs::write(&path, &output_bytes).unwrap_or_else(|e| {
                eprintln!("error: failed to write {:?}: {e}", path);
                std::process::exit(1);
            });
        }
        None => {
            use std::io::Write;
            std::io::stdout().write_all(&output_bytes).unwrap();
            if matches!(cli.format, OutputFormat::Ron) {
                println!();
            }
        }
    }
}

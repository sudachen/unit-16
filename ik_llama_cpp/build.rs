//
// slightly modified llama-cpp-sys-2 build script
//


use cmake::Config;
use glob::glob;
use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;
use walkdir::DirEntry;

enum WindowsVariant {
    Msvc,
    Other,
}

enum AppleVariant {
    MacOS,
    Other,
}

enum TargetOs {
    Windows(WindowsVariant),
    Apple(AppleVariant),
    Linux,
}

macro_rules! debug_log {
    ($($arg:tt)*) => {
        if std::env::var("BUILD_DEBUG").is_ok() {
            println!("cargo:warning=[DEBUG] {}", format!($($arg)*));
        }
    };
}

fn parse_target_os() -> Result<(TargetOs, String), String> {
    let target = env::var("TARGET").unwrap();

    if target.contains("windows") {
        if target.ends_with("-windows-msvc") {
            Ok((TargetOs::Windows(WindowsVariant::Msvc), target))
        } else {
            Ok((TargetOs::Windows(WindowsVariant::Other), target))
        }
    } else if target.contains("apple") {
        if target.ends_with("-apple-darwin") {
            Ok((TargetOs::Apple(AppleVariant::MacOS), target))
        } else {
            Ok((TargetOs::Apple(AppleVariant::Other), target))
        }
    } else if target.contains("linux") {
        Ok((TargetOs::Linux, target))
    } else {
        Err(target)
    }
}

fn get_cargo_target_dir() -> Result<PathBuf, Box<dyn std::error::Error>> {
    let out_dir = env::var("OUT_DIR")?;
    let path = PathBuf::from(out_dir);
    let target_dir = path
        .ancestors()
        .nth(3)
        .ok_or("OUT_DIR is not deep enough")?;
    Ok(target_dir.to_path_buf())
}

fn extract_lib_names(out_dir: &Path, build_shared_libs: bool, target_os: &TargetOs) -> Vec<String> {
    let lib_pattern = match target_os {
        TargetOs::Windows(_) => "*.lib",
        TargetOs::Apple(_) => {
            if build_shared_libs {
                "*.dylib"
            } else {
                "*.a"
            }
        }
        TargetOs::Linux  => {
            if build_shared_libs {
                "*.so"
            } else {
                "*.a"
            }
        }
    };
    let libs_dir = out_dir.join("lib*");
    let pattern = libs_dir.join(lib_pattern);
    debug_log!("Extract libs {}", pattern.display());

    let mut lib_names: Vec<String> = Vec::new();

    // Process the libraries based on the pattern
    for entry in glob(pattern.to_str().unwrap()).unwrap() {
        match entry {
            Ok(path) => {
                let stem = path.file_stem().unwrap();
                let stem_str = stem.to_str().unwrap();

                // Remove the "lib" prefix if present
                let lib_name = if stem_str.starts_with("lib") {
                    stem_str.strip_prefix("lib").unwrap_or(stem_str)
                } else {
                    if path.extension() == Some(std::ffi::OsStr::new("a")) {
                        let target = path.parent().unwrap().join(format!("lib{}.a", stem_str));
                        std::fs::rename(&path, &target).unwrap_or_else(|e| {
                            panic!("Failed to rename {path:?} to {target:?}: {e:?}");
                        })
                    }
                    stem_str
                };
                lib_names.push(lib_name.to_string());
            }
            Err(e) => println!("cargo:warning=error={}", e),
        }
    }
    lib_names
}

fn extract_lib_assets(out_dir: &Path, target_os: &TargetOs) -> Vec<PathBuf> {
    let shared_lib_pattern = match target_os {
        TargetOs::Windows(_) => "*.dll",
        TargetOs::Apple(_) => "*.dylib",
        TargetOs::Linux => "*.so",
    };

    let shared_libs_dir = match target_os {
        TargetOs::Windows(_) => "bin",
        _ => "lib",
    };
    let libs_dir = out_dir.join(shared_libs_dir);
    let pattern = libs_dir.join(shared_lib_pattern);
    debug_log!("Extract lib assets {}", pattern.display());
    let mut files = Vec::new();

    for entry in glob(pattern.to_str().unwrap()).unwrap() {
        match entry {
            Ok(path) => {
                files.push(path);
            }
            Err(e) => eprintln!("cargo:warning=error={}", e),
        }
    }

    files
}

fn macos_link_search_path() -> Option<String> {
    let output = Command::new("clang")
        .arg("--print-search-dirs")
        .output()
        .ok()?;
    if !output.status.success() {
        println!(
            "failed to run 'clang --print-search-dirs', continuing without a link search path"
        );
        return None;
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    for line in stdout.lines() {
        if line.contains("libraries: =") {
            let path = line.split('=').nth(1)?;
            return Some(format!("{}/lib/darwin", path));
        }
    }

    println!("failed to determine link search path, continuing without it");
    None
}

fn is_hidden(e: &DirEntry) -> bool {
    e.file_name()
        .to_str()
        .map(|s| s.starts_with('.'))
        .unwrap_or_default()
}

fn main() {
    println!("cargo:rerun-if-changed=build.rs");

    let (target_os, target_triple) =
        parse_target_os().unwrap_or_else(|t| panic!("Failed to parse target os {t}"));
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());

    let target_dir = get_cargo_target_dir().unwrap();
    let manifest_dir = env::var("CARGO_MANIFEST_DIR").expect("Failed to get CARGO_MANIFEST_DIR");
    let llama_src = Path::new(&manifest_dir).join("ik_llama.cpp");
    let build_shared_libs = cfg!(feature = "dynamic-link");

    let build_shared_libs = std::env::var("LLAMA_BUILD_SHARED_LIBS")
        .map(|v| v == "1")
        .unwrap_or(build_shared_libs);
    let profile = env::var("LLAMA_LIB_PROFILE").unwrap_or("Release".to_string());
    let static_crt = env::var("LLAMA_STATIC_CRT")
        .map(|v| v == "1")
        .unwrap_or(false);

    println!("cargo:rerun-if-env-changed=LLAMA_LIB_PROFILE");
    println!("cargo:rerun-if-env-changed=LLAMA_BUILD_SHARED_LIBS");
    println!("cargo:rerun-if-env-changed=LLAMA_STATIC_CRT");

    debug_log!("TARGET: {}", target_triple);
    debug_log!("CARGO_MANIFEST_DIR: {}", manifest_dir);
    debug_log!("TARGET_DIR: {}", target_dir.display());
    debug_log!("OUT_DIR: {}", out_dir.display());
    debug_log!("BUILD_SHARED: {}", build_shared_libs);

    // Make sure that changes to the llama.cpp project trigger a rebuild.
    let rebuild_on_children_of = [
        llama_src.join("src"),
        llama_src.join("ggml/src"),
        llama_src.join("common"),
    ];
    for entry in walkdir::WalkDir::new(&llama_src)
        .into_iter()
        .filter_entry(|e| !is_hidden(e))
    {
        let entry = entry.expect("Failed to obtain entry");
        let rebuild = entry
            .file_name()
            .to_str()
            .map(|f| f.starts_with("CMake"))
            .unwrap_or_default()
            || rebuild_on_children_of
            .iter()
            .any(|src_folder| entry.path().starts_with(src_folder));
        if rebuild {
            println!("cargo:rerun-if-changed={}", entry.path().display());
        }
    }

    // Speed up build
    unsafe {env::set_var(
        "CMAKE_BUILD_PARALLEL_LEVEL",
        std::thread::available_parallelism()
            .unwrap()
            .get()
            .to_string(),
    );}

    // Bindings
    let mut bindings_builder = bindgen::Builder::default()
        .header("wrapper.h")
        .clang_arg(format!("-I{}", llama_src.join("include").display()))
        .clang_arg(format!("-I{}", llama_src.join("ggml/include").display()))
        .clang_arg(format!("-I{}", llama_src.join("src").display()))
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .derive_partialeq(true)
        .allowlist_function("ggml_.*")
        .allowlist_type("ggml_.*")
        .allowlist_function("gguf_.*")
        .allowlist_type("gguf_.*")
        .allowlist_function("llama_.*")
        .allowlist_type("llama_.*")
        .allowlist_function("llama_rs_.*")
        .allowlist_type("llama_rs_.*")
        .prepend_enum_name(false);

    // Fix bindgen header discovery on Windows MSVC
    // Use cc crate to discover MSVC include paths by compiling a dummy file
    if matches!(target_os, TargetOs::Windows(WindowsVariant::Msvc)) {
        // Create a minimal dummy C file to extract compiler flags
        let out_dir = env::var("OUT_DIR").unwrap();
        let dummy_c = Path::new(&out_dir).join("dummy.c");
        std::fs::write(&dummy_c, "int main() { return 0; }").unwrap();

        // Use cc crate to get compiler with proper environment setup
        let mut build = cc::Build::new();
        build.file(&dummy_c);

        // Get the actual compiler command cc would use
        let compiler = build.try_get_compiler().unwrap();

        // Extract include paths by checking compiler's environment
        // cc crate sets up MSVC environment internally
        let env_include = compiler
            .env()
            .iter()
            .find(|(k, _)| k.eq_ignore_ascii_case("INCLUDE"))
            .map(|(_, v)| v);

        if let Some(include_paths) = env_include {
            for include_path in include_paths
                .to_string_lossy()
                .split(';')
                .filter(|s| !s.is_empty())
            {
                bindings_builder = bindings_builder
                    .clang_arg("-isystem")
                    .clang_arg(include_path);
                debug_log!("Added MSVC include path: {}", include_path);
            }
        }

        // Add MSVC compatibility flags
        bindings_builder = bindings_builder
            .clang_arg(format!("--target={}", target_triple))
            .clang_arg("-fms-compatibility")
            .clang_arg("-fms-extensions");

        debug_log!(
            "Configured bindgen with MSVC toolchain for target: {}",
            target_triple
        );
    }
    let bindings = bindings_builder
        .generate()
        .expect("Failed to generate bindings");

    // Write the generated bindings to an output file
    let bindings_path = out_dir.join("bindings.rs");
    bindings
        .write_to_file(bindings_path)
        .expect("Failed to write bindings");

    println!("cargo:rerun-if-changed=wrapper.h");
    println!("cargo:rerun-if-changed=wrapper_common.h");
    println!("cargo:rerun-if-changed=wrapper_common.cpp");
    debug_log!("Bindings Created");

    let mut common_wrapper_build = cc::Build::new();
    common_wrapper_build
        .cpp(true)
        .file("wrapper_common.cpp")
        .include(&llama_src)
        .include(llama_src.join("common"))
        .include(llama_src.join("include"))
        .include(llama_src.join("ggml/include"))
        .include(llama_src.join("src"))
        .include(llama_src.join("vendor"))
        .flag_if_supported("-std=c++17")
        .pic(true);

    if matches!(target_os, TargetOs::Windows(WindowsVariant::Msvc)) {
        common_wrapper_build.flag("/std:c++17");
    }

    // When static-stdcxx is enabled on Android, suppress the cc crate's automatic
    // C++ stdlib linking (which defaults to c++_shared) so we can link c++_static instead.
    common_wrapper_build.compile("ik_llama_cpp_common_wrapper");

    // Build with Cmake

    let mut config = Config::new(&llama_src);

    // Would require extra source files to pointlessly
    // be included in what's uploaded to and downloaded from
    // crates.io, so deactivating these instead
    config.define("LLAMA_BUILD_TESTS", "OFF");
    config.define("LLAMA_BUILD_EXAMPLES", "OFF");
    config.define("LLAMA_BUILD_SERVER", "OFF");
    config.define("LLAMA_BUILD_TOOLS", "OFF");
    config.define("LLAMA_BUILD_COMMON", "ON");
    config.define("LLAMA_CURL", "OFF");

    // Pass CMAKE_ environment variables down to CMake
    for (key, value) in env::vars() {
        if key.starts_with("CMAKE_") {
            config.define(&key, &value);
        }
    }

    // extract the target-cpu config value, if specified
    let target_cpu = std::env::var("CARGO_ENCODED_RUSTFLAGS")
        .ok()
        .and_then(|rustflags| {
            rustflags
                .split('\x1f')
                .find(|f| f.contains("target-cpu="))
                .and_then(|f| f.split("target-cpu=").nth(1))
                .map(|s| s.to_string())
        });

    if target_cpu == Some("native".into()) {
        debug_log!("Detected target-cpu=native, compiling with GGML_NATIVE");
        config.define("GGML_NATIVE", "ON");
    }
    // if native isn't specified, enable specific features for ggml instead
    else {
        // rust code isn't using `target-cpu=native`, so llama.cpp shouldn't use GGML_NATIVE either
        config.define("GGML_NATIVE", "OFF");

        // if `target-cpu` is set set, also set -march for llama.cpp to the same value
        if let Some(ref cpu) = target_cpu {
            debug_log!("Setting baseline architecture: -march={}", cpu);
            config.cflag(format!("-march={}", cpu));
            config.cxxflag(format!("-march={}", cpu));
        }

        // I expect this env var to always be present
        let features = std::env::var("CARGO_CFG_TARGET_FEATURE")
            .expect("Env var CARGO_CFG_TARGET_FEATURE not found.");
        debug_log!("Compiling with target features: {}", features);

        // list of rust target_features here:
        //   https://doc.rust-lang.org/reference/attributes/codegen.html#the-target_feature-attribute
        // GGML config flags have been found by looking at:
        //   llama.cpp/ggml/src/ggml-cpu/CMakeLists.txt
        for feature in features.split(',') {
            match feature {
                "avx" => {
                    config.define("GGML_AVX", "ON");
                }
                "avx2" => {
                    config.define("GGML_AVX2", "ON");
                }
                "avx512bf16" => {
                    config.define("GGML_AVX512_BF16", "ON");
                }
                "avx512vbmi" => {
                    config.define("GGML_AVX512_VBMI", "ON");
                }
                "avx512vnni" => {
                    config.define("GGML_AVX512_VNNI", "ON");
                }
                "avxvnni" => {
                    config.define("GGML_AVX_VNNI", "ON");
                }
                "bmi2" => {
                    config.define("GGML_BMI2", "ON");
                }
                "f16c" => {
                    config.define("GGML_F16C", "ON");
                }
                "fma" => {
                    config.define("GGML_FMA", "ON");
                }
                "sse4.2" => {
                    config.define("GGML_SSE42", "ON");
                }
                _ => {
                    debug_log!(
                        "Unrecognized cpu feature: '{}' - skipping GGML config for it.",
                        feature
                    );
                    continue;
                }
            };
        }
    }

    config.define(
        "BUILD_SHARED_LIBS",
        if build_shared_libs { "ON" } else { "OFF" },
    );

    if matches!(target_os, TargetOs::Apple(_)) {
        config.define("GGML_BLAS", "OFF");
    }

    if (matches!(target_os, TargetOs::Windows(WindowsVariant::Msvc))
        && matches!(
            profile.as_str(),
            "Release" | "RelWithDebInfo" | "MinSizeRel"
        ))
    {
        // Debug Rust builds under MSVC turn off optimization even though we're ideally building the release profile of llama.cpp.
        // Looks like an upstream bug:
        // https://github.com/rust-lang/cmake-rs/issues/240
        // For now explicitly reinject the optimization flags that a CMake Release build is expected to have on in this scenario.
        // This fixes CPU inference performance when part of a Rust debug build.
        for flag in &["/O2", "/DNDEBUG", "/Ob2"] {
            config.cflag(flag);
            config.cxxflag(flag);
        }
    }

    config.static_crt(static_crt);

    if matches!(target_os, TargetOs::Linux)
        && target_triple.contains("aarch64")
        && target_cpu != Some("native".into())
    {
        // If the target-cpu is not specified as native, we take off the native ARM64 support.
        // It is useful in docker environments where the native feature is not enabled.
        config.define("GGML_NATIVE", "OFF");
        config.define("GGML_CPU_ARM_ARCH", "armv8-a");
    }

    if cfg!(feature = "cuda") {
        config.define("GGML_CUDA", "ON");
    }

    if cfg!(feature = "openmp") {
        config.define("GGML_OPENMP", "ON");
        println!("cargo:rustc-link-lib=gomp");
    }

    // General
    config
        .profile(&profile)
        .very_verbose(std::env::var("CMAKE_VERBOSE").is_ok()) // Not verbose by default
        .always_configure(false);

    let build_dir = config.build();

    // Search paths
    println!("cargo:rustc-link-search={}", out_dir.join("lib").display());
    println!(
        "cargo:rustc-link-search={}",
        out_dir.join("lib64").display()
    );
    println!("cargo:rustc-link-search={}", build_dir.display());

    if cfg!(feature = "cuda") && !build_shared_libs {
        // Re-run build script if CUDA_PATH environment variable changes
        println!("cargo:rustc-link-lib=cuda");
        println!("cargo:rerun-if-env-changed=CUDA_PATH");

        // Add CUDA library directories to the linker search path
        for lib_dir in find_cuda_helper::find_cuda_lib_dirs() {
            println!("cargo:rustc-link-search=native={}", lib_dir.display());
        }
        println!("cargo:rustc-link-lib=cudart"); // Links to cudart64_*.dll
        println!("cargo:rustc-link-lib=cublas"); // Links to cublas64_*.dll
        println!("cargo:rustc-link-lib=cublasLt"); // Links to cublasLt64_*.dll
    }

    // Link libraries
    let llama_libs_kind = if build_shared_libs
    {
        "dylib"
    } else {
        "static"
    };

    let llama_libs = extract_lib_names(&out_dir, build_shared_libs, &target_os);

    assert_ne!(llama_libs.len(), 0);

    let common_lib_dir = out_dir.join("build").join("common");
    if common_lib_dir.is_dir() {
        println!(
            "cargo:rustc-link-search=native={}",
            common_lib_dir.display()
        );
        let common_profile_dir = common_lib_dir.join(&profile);
        if common_profile_dir.is_dir() {
            println!(
                "cargo:rustc-link-search=native={}",
                common_profile_dir.display()
            );
        }
        println!("cargo:rustc-link-lib=static=common");
    }

    for lib in llama_libs {
        let link = format!("cargo:rustc-link-lib={}={}", llama_libs_kind, lib);
        debug_log!("LINK {link}",);
        println!("{link}",);
    }


    match target_os {
        TargetOs::Windows(WindowsVariant::Msvc) => {
            println!("cargo:rustc-link-lib=advapi32");
            let crt_static = env::var("CARGO_CFG_TARGET_FEATURE")
                .unwrap_or_default()
                .contains("crt-static");
            if cfg!(debug_assertions) {
                if crt_static {
                    println!("cargo:rustc-link-lib=libcmtd");
                } else {
                    println!("cargo:rustc-link-lib=dylib=msvcrtd");
                }
            }
        }
        TargetOs::Linux => {
            println!("cargo:rustc-link-lib=dylib=stdc++");
        }
        TargetOs::Apple(ref variant) => {
            println!("cargo:rustc-link-lib=framework=Foundation");
            println!("cargo:rustc-link-lib=framework=Metal");
            println!("cargo:rustc-link-lib=framework=MetalKit");
            println!("cargo:rustc-link-lib=framework=Accelerate");
            println!("cargo:rustc-link-lib=c++");

            match variant {
                AppleVariant::MacOS => {
                    // On (older) OSX we need to link against the clang runtime,
                    // which is hidden in some non-default path.
                    //
                    // More details at https://github.com/alexcrichton/curl-rust/issues/279.
                    if let Some(path) = macos_link_search_path() {
                        println!("cargo:rustc-link-lib=clang_rt.osx");
                        println!("cargo:rustc-link-search={}", path);
                    }
                }
                AppleVariant::Other => (),
            }
        }
        _ => (),
    }

    // copy DLLs to target
    if build_shared_libs {
        let libs_assets = extract_lib_assets(&out_dir, &target_os);
        for asset in libs_assets {
            let asset_clone = asset.clone();
            let filename = asset_clone.file_name().unwrap();
            let filename = filename.to_str().unwrap();
            let dst = target_dir.join(filename);
            debug_log!("HARD LINK {} TO {}", asset.display(), dst.display());
            if !dst.exists() {
                std::fs::hard_link(asset.clone(), dst).unwrap();
            }

            // Copy DLLs to examples as well
            if target_dir.join("examples").exists() {
                let dst = target_dir.join("examples").join(filename);
                debug_log!("HARD LINK {} TO {}", asset.display(), dst.display());
                if !dst.exists() {
                    std::fs::hard_link(asset.clone(), dst).unwrap();
                }
            }

            // Copy DLLs to target/profile/deps as well for tests
            let dst = target_dir.join("deps").join(filename);
            debug_log!("HARD LINK {} TO {}", asset.display(), dst.display());
            if !dst.exists() {
                std::fs::hard_link(asset.clone(), dst).unwrap();
            }
        }
    }
}


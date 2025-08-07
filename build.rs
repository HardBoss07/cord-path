fn main() {
    let out_dir = std::env::var("OUT_DIR").unwrap();

    cc::Build::new()
        .cuda(true)
        .file("cuda/kernel.cu")
        .compile("cuda_kernels");

    println!("cargo:rustc-link-search=native={}", out_dir);
    println!("cargo:rustc-link-lib=static=cuda_kernels");
}

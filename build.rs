fn main() {
    cc::Build::new()
        .cuda(true)
        .file("cuda/kernel.cu")
        .compile("cuda_kernels");
}
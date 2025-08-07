use clap::Parser;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::PathBuf;

#[link(name = "cuda_kernels", kind = "static")]
extern "C" {
    fn calculate_distances(xs: *const i32, ys: *const i32, distances: *mut f32, n: i32);
}

#[derive(Debug)]
struct Point {
    x: i32,
    y: i32,
}

#[derive(Parser, Debug)]
#[command(name = "cord-path", version, about = "Coordinate Path Planner")]
struct Args {
    #[arg(short, long)]
    file: PathBuf,

    #[arg(short = 'x')]
    start_x: Option<i32>,

    #[arg(short = 'y')]
    start_y: Option<i32>,
}

fn load_points_from_csv(path: &PathBuf) -> std::io::Result<Vec<Point>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);

    let mut points = Vec::new();

    for (line_num, line) in reader.lines().enumerate() {
        let line = line?;
        let values: Vec<&str> = line.trim().split(',').collect();

        if values.len() != 2 {
            eprintln!("Warning: Invalid line {}: '{}'", line_num + 1, line);
            continue;
        }

        let x = values[0].trim().parse::<i32>();
        let y = values[1].trim().parse::<i32>();

        match (x, y) {
            (Ok(x), Ok(y)) => points.push(Point { x, y }),
            _ => eprintln!("Warning: Could not parse line {}: '{}'", line_num + 1, line),
        }
    }

    Ok(points)
}

fn call_cuda_distance(points: &[Point]) {
    let n = points.len();
    let xs: Vec<i32> = points.iter().map(|p| p.x).collect();
    let ys: Vec<i32> = points.iter().map(|p| p.y).collect();

    let mut distances = vec![0.0f32; n];

    unsafe {
        calculate_distances(
            xs.as_ptr(),
            ys.as_ptr(),
            distances.as_mut_ptr(),
            n as i32,
        );
    }

    println!("Distances from first point:");
    for (i, d) in distances.iter().enumerate() {
        println!("  to point {:>3}: {:.2}", i, d);
    }
}

fn main() {
    let args = Args::parse();

    println!("Reading file: {:?}", args.file);
    let points = load_points_from_csv(&args.file).expect("Failed to load points");

    if points.len() > 1 {
        call_cuda_distance(&points);
    }

    if let Some(x) = args.start_x {
        if let Some(y) = args.start_y {
            println!("Start position: ({}, {})", x, y);
        } else {
            eprintln!("Start Y missing when X is provided.");
        }
    }
}
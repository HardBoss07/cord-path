use clap::Parser;
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;

#[link(name = "cuda_kernels", kind = "static")]
extern "C" {
    fn compute_distance_matrix(xs: *const i32, ys: *const i32, dist_matrix: *mut f32, n: i32);
}

#[derive(Debug)]
struct Point {
    x: i32,
    y: i32,
}

#[derive(Parser, Debug)]
#[command(name = "cord-path", version, about = "Coordinate Path Planner")]
struct Args {
    /// Path to CSV file with point coordinates
    #[arg(short, long, value_name = "FILE")]
    file: PathBuf,

    /// Optional starting X coordinate
    #[arg(short = 'x', long, value_name = "X", allow_hyphen_values = true)]
    start_x: Option<i32>,

    /// Optional starting Y coordinate
    #[arg(short = 'y', long, value_name = "Y", allow_hyphen_values = true)]
    start_y: Option<i32>,

    /// Optional output file path (CSV)
    #[arg(short = 'o', long, value_name = "OUTPUT_FILE")]
    output: Option<PathBuf>,

    /// Suppress all console output (only works with --output)
    #[arg(short, long, action = clap::ArgAction::SetTrue)]
    quiet: bool,
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

fn call_cuda_distance_matrix(points: &[Point]) -> Vec<f32> {
    let n = points.len();
    let xs: Vec<i32> = points.iter().map(|p| p.x).collect();
    let ys: Vec<i32> = points.iter().map(|p| p.y).collect();

    let mut dist_matrix = vec![0.0f32; n * n];

    unsafe {
        compute_distance_matrix(
            xs.as_ptr(),
            ys.as_ptr(),
            dist_matrix.as_mut_ptr(),
            n as i32,
        );
    }

    dist_matrix
}

// Nearest Neighbor heuristic
fn nearest_neighbor_tsp(dist_matrix: &[f32], n: usize, start: usize) -> Vec<usize> {
    let mut visited = vec![false; n];
    let mut path = Vec::with_capacity(n);
    let mut current = start;

    path.push(current);
    visited[current] = true;

    for _ in 1..n {
        let mut next = None;
        let mut min_dist = f32::MAX;

        for candidate in 0..n {
            if !visited[candidate] && dist_matrix[current * n + candidate] < min_dist {
                min_dist = dist_matrix[current * n + candidate];
                next = Some(candidate);
            }
        }

        if let Some(next_node) = next {
            path.push(next_node);
            visited[next_node] = true;
            current = next_node;
        } else {
            break;
        }
    }

    path
}

// 2-opt local optimization
fn two_opt(dist_matrix: &[f32], n: usize, path: &mut Vec<usize>) {
    let mut improved = true;

    while improved {
        improved = false;
        for i in 1..(n - 1) {
            for j in (i + 1)..n {
                let a = path[i - 1];
                let b = path[i];
                let c = path[j - 1];
                let d = path[j];

                let old_dist = dist_matrix[a * n + b] + dist_matrix[c * n + d];
                let new_dist = dist_matrix[a * n + c] + dist_matrix[b * n + d];

                if new_dist < old_dist {
                    path[i..j].reverse();
                    improved = true;
                }
            }
        }
    }
}

// Calculate total path length
fn path_length(dist_matrix: &[f32], path: &[usize]) -> f32 {
    let n = path.len();
    let mut sum = 0.0;
    for i in 1..n {
        sum += dist_matrix[path[i - 1] * n + path[i]];
    }
    sum
}

// Write the output to a file if specified
fn write_output(path: &PathBuf, points: &[Point], path_order: &[usize]) -> std::io::Result<()> {
    let mut file = File::create(path)?;
    for &idx in path_order {
        let p = &points[idx];
        writeln!(file, "{}, {}", p.x, p.y)?;
    }
    Ok(())
}

fn main() {
    let args = Args::parse();

    // Disallow quiet without output file
    if args.quiet && args.output.is_none() {
        eprintln!("Error: --quiet can only be used with --output");
        std::process::exit(1);
    }

    let mut points = load_points_from_csv(&args.file).expect("Failed to load points");
    let n_original = points.len();

    if n_original == 0 {
        eprintln!("No points loaded.");
        return;
    }

    // Insert start position if provided
    if let (Some(sx), Some(sy)) = (args.start_x, args.start_y) {
        points.insert(0, Point { x: sx, y: sy });
    }

    let n = points.len();
    let dist_matrix = call_cuda_distance_matrix(&points);
    let start_index = 0; // start position at index 0 if inserted

    let mut path = nearest_neighbor_tsp(&dist_matrix, n, start_index);

    if !args.quiet {
        println!(
            "Path length before 2-opt: {}",
            path_length(&dist_matrix, &path)
        );
    }

    two_opt(&dist_matrix, n, &mut path);

    if !args.quiet {
        println!(
            "Path length after 2-opt: {}",
            path_length(&dist_matrix, &path)
        );
        println!("Path order:");
        for &idx in &path {
            let p = &points[idx];
            println!("Point {}: ({}, {})", idx, p.x, p.y);
        }
    }

    if let Some(output_path) = args.output {
        write_output(&output_path, &points, &path).expect("Failed to write output");
    }
}

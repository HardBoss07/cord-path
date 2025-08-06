use clap::Parser;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::PathBuf;

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


fn main() {
    let args = Args::parse();

    println!("Reading file: {:?}", args.file);
    let points = load_points_from_csv(&args.file).expect("Failed to load points");

    if let Some(x) = args.start_x {
        if let Some(y) = args.start_y {
            println!("Start position: ({}, {})", x, y);
        } else {
            eprintln!("Start Y missing when X is provided.");
        }
    }

    println!("Loaded {} points:", points.len());
    for (i, p) in points.iter().enumerate() {
        println!("{:>3}. ({}, {})", i + 1, p.x, p.y);
    }
}
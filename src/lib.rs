//! This is documentation for the `foo` crate.
//!
//! The foo crate is meant to be used for bar.

/// That is not dead whcih can eternal lie
/// `this is a code thing`
pub mod delaunay {
	//! Something?
use std::ops::Add;
use std::ops::Sub;
use std::collections::HashMap;
use std::collections::HashSet;
#[inline]
fn next_idx(n: usize, m: usize) -> usize {
	if n < (m-1) { n + 1 } else { 0 }
}
#[test]
fn test_next_idx() {
	assert_eq!(1, next_idx(0, 3));
	assert_eq!(2, next_idx(1, 3));
	assert_eq!(0, next_idx(2, 3));
}
#[inline]
fn prev_idx(n: usize, m: usize) -> usize {
	if n == 0 { m - 1 } else { n - 1 }
}
#[test]
fn test_prev_idx() {
	assert_eq!(2, prev_idx(0, 3));
	assert_eq!(0, prev_idx(1, 3));
	assert_eq!(1, prev_idx(2, 3));
}
	/// Represents an (X, Y) coordinate
	#[derive(Debug, Clone, Copy)]
	pub struct Point {
		x: f64,
		y: f64
	}

	impl Point {
		pub fn new(x: f64, y: f64) -> Point {
			Point{ x: x, y: y }
		}
		fn mag(&self) -> f64 {
			self.x*self.x + self.y*self.y
		}
	}

	impl Add for Point {
		type Output = Point;
		fn add(self, other: Point) -> Point {
			Point { x: self.x + other.x, y: self.y + other.y }
		}
	}

	impl Sub for Point {
		type Output = Point;
		fn sub(self, other: Point) -> Point {
			Point { x: self.x - other.x, y: self.y - other.y }
		}
	}

	// Opposite triangles for each face
	#[derive(Debug, Clone, Copy, PartialEq)]
	pub struct TStruct(pub Option<Triangle>, pub Option<Triangle>, pub Option<Triangle>);

	impl TStruct {
		fn get(&self, n: usize) -> Option<Triangle> {
			match n {
				0 => self.0,
				1 => self.1,
				2 => self.2,
				_ => None
			}
		}
		fn get_ccw_op(&self, t: Triangle) -> usize {
			match *self {
				TStruct(Some(x), _, _) if x == t => 0,
				TStruct(_, Some(x), _) if x == t => 1,
				TStruct(_, _, Some(x)) if x == t => 2,
				_ => panic!("At the Disco")
			}
		}

		fn update_with_neighbour(self, e0: usize, e1: usize, t: Triangle) -> TStruct {
			match self {
				TStruct(Some(x), b, c) if x.has_edges(e0, e1) => TStruct(Some(t), b, c),
				TStruct(a, Some(x), c) if x.has_edges(e0, e1) => TStruct(a, Some(t), c),
				TStruct(a, b, Some(x)) if x.has_edges(e0, e1) => TStruct(a, b, Some(t)),
				_ => self
			}
		}
	}

	#[test]
	fn test_update_with_neighbour() {
		let ts = TStruct(None, Some(Triangle(4, 0, 1)), Some(Triangle(4, 2, 3)));
		let new_ts = ts.update_with_neighbour(3, 4, Triangle(5,3,4));
		let expected_ts = TStruct(None, Some(Triangle(4,0,1)), Some(Triangle(5,3,4)));
		assert_eq!(expected_ts, new_ts);

		let ts = TStruct(None, Some(Triangle(4, 1, 2)), Some(Triangle(4, 3, 0)));
		let new_ts = ts.update_with_neighbour(4, 1, Triangle(5,4,1));
		let expected_ts = TStruct(None, Some(Triangle(5,4,1)), Some(Triangle(4,3,0)));
		assert_eq!(expected_ts, new_ts);

		let ts = TStruct(None, Some(Triangle(5, 3, 4)), Some(Triangle(5, 1, 2)));
		let new_ts = ts.update_with_neighbour(2, 5, Triangle(6,2,5));
		let expected_ts = TStruct(None, Some(Triangle(5,3,4)), Some(Triangle(6,2,5)));
		assert_eq!(expected_ts, new_ts);

		let ts = TStruct(Some(Triangle(4,3,0)), Some(Triangle(5, 4, 1)), Some(Triangle(5, 2, 3)));
		let new_ts = ts.update_with_neighbour(5, 4, Triangle(6,5,4));
		let expected_ts = TStruct(Some(Triangle(4,3,0)), Some(Triangle(6,5,4)), Some(Triangle(5,2,3)));
		assert_eq!(expected_ts, new_ts);

		let ts = TStruct(None, Some(Triangle(4, 0, 1)), Some(Triangle(5, 3, 4)));
		let new_ts = ts.update_with_neighbour(4, 0, Triangle(6,4,0));
		let expected_ts = TStruct(None, Some(Triangle(6,4,0)), Some(Triangle(5,3,4)));
		assert_eq!(expected_ts, new_ts);

		let ts = TStruct(None, Some(Triangle(5, 3, 4)), Some(Triangle(6, 2, 5)));
		let new_ts = ts.update_with_neighbour(2, 5, Triangle(7,2,5));
		let expected_ts = TStruct(None, Some(Triangle(5,3,4)), Some(Triangle(7,2,5)));
		assert_eq!(expected_ts, new_ts);

		let ts = TStruct(Some(Triangle(5, 3, 4)), Some(Triangle(6, 4, 0)), Some(Triangle(6,2,5)));
		let new_ts = ts.update_with_neighbour(5, 6, Triangle(7, 5, 6));
		let expected_ts = TStruct(Some(Triangle(5, 3, 4)), Some(Triangle(6, 4, 0)), Some(Triangle(7,5,6)));
		assert_eq!(expected_ts, new_ts);

		let ts = TStruct(Some(Triangle(4, 3, 0)), Some(Triangle(6, 0, 1)), Some(Triangle(6,5,4)));
		let new_ts = ts.update_with_neighbour(6, 0, Triangle(7, 6, 0));
		let expected_ts = TStruct(Some(Triangle(4, 3, 0)), Some(Triangle(7, 6, 0)), Some(Triangle(6, 5, 4)));
		assert_eq!(expected_ts, new_ts);
	}

	#[derive(Debug)]
	pub struct Delaunay2D {
		pub coords: Vec<Point>,
		pub triangles: HashMap<Triangle, TStruct>,
		pub circles: HashMap<Triangle, (Point, f64)>
	}

	#[derive(Debug, PartialEq, Eq, Hash, Clone, Copy)]
	pub struct Triangle(pub usize, pub usize, pub usize);

	impl Triangle {
		fn get(&self, n: usize) -> usize {
			match n {
				0 => self.0,
				1 => self.1,
				2 => self.2,
				_ => panic!("Triangles only have three sides")
			}
		}

		fn has_edges(&self, e0: usize, e1: usize) -> bool {
			(self.0 == e0 || self.1 == e0 || self.2 == e0) &&
				(self.0 == e1 || self.1 == e1 || self.2 == e1)
		}
	}

	impl Delaunay2D {
		pub fn new(center: Point, radius: f64) -> Delaunay2D {
			let coords = vec!(
				Point { x: center.x - radius, y: center.y - radius },
				Point { x: center.x + radius, y: center.y - radius },
				Point { x: center.x + radius, y: center.y + radius },
				Point { x: center.x - radius, y: center.y + radius });

			let mut triangles = HashMap::new();
			let mut circles = HashMap::new();
			let t1 = Triangle(0, 1, 3);
			let t2 = Triangle(2, 3, 1);
			triangles.insert(t1, TStruct(Some(t2), None, None));
			triangles.insert(t2, TStruct(Some(t1), None, None));
			circles.insert(t1, t1.circumcenter(&coords));
			circles.insert(t2, t2.circumcenter(&coords));
			Delaunay2D { coords: coords, triangles: triangles, circles: circles }
		}

		fn in_circle_fast(&self, tri: Triangle, p: Point) -> bool {
			let (center, radius) = self.circles[&tri];
			(center - p).mag() <= radius
		}

		// fn in_circle_robust(&self, tri: Triangle, p: Point) -> bool {
		// 	let (a, b, c) = (self.coords[tri.0] - p, self.coords[tri.1] - p, self.coords[tri.2] - p);
		// 	let a_mag = a.mag();
		// 	let b_mag = b.mag();
		// 	let c_mag = c.mag();
		// 	let det = a.x * (b.y * c_mag - b_mag * c.y)
		// 	        + a.y * (b_mag * c.x - c_mag * b.x)
		// 	        + a_mag * (b.x * c.y - c.x * b.y);

		// 	det > 0f64
		// }
		
		#[allow(while_true)]
		pub fn add_point(&mut self, p: Point) {
			let idx = self.coords.len();
			self.coords.push(p);

			let bad_triangles: HashSet<_> = self.triangles.keys().cloned().filter(|&t| self.in_circle_fast(t, p)).collect();
			
			let mut boundary: Vec<(usize, usize, Option<Triangle>)> = vec!();
			let mut t: Triangle = *bad_triangles.iter().next().unwrap();
			let mut edge = 0;

			while true {
				// Check if edge of triangle T is on the boundary...
	            // if opposite triangle of this edge is external to the list
				let tri_op = self.triangles[&t].get(edge);
				if tri_op.is_none() || !bad_triangles.contains(&tri_op.unwrap()) {

                	// Insert edge and external triangle into boundary list
					boundary.push((t.get(next_idx(edge, 3)), t.get(prev_idx(edge, 3)), tri_op));
                	// Move to next CCW edge in this triangle
					edge = next_idx(edge, 3);
					if boundary[0].0 == boundary[boundary.len() - 1].1 {
						break;
					}
				} else if let Some(tri_op) = tri_op {
					// Move to next CCW edge in opposite triangle
					let ccw_op = self.triangles[&tri_op].get_ccw_op(t);
					//edge = (self.triangles[tri_op].index(T) + 1) % 3
					edge = next_idx(ccw_op, 3);
					t = tri_op;
				}
			}
			for t in bad_triangles {
				self.triangles.remove(&t);
				self.circles.remove(&t);
			}
			// Retriangle the hole left by bad_triangles
			let mut new_triangles: Vec<Triangle> = vec!();
			for (e0, e1, tri_op) in boundary {
            	// Create a new triangle using point p and edge extremes
				let t = Triangle(idx, e0, e1);
             	// Store circumcenter and circumradius of the triangle
				self.circles.insert(t, t.circumcenter(&self.coords));
             	// Set opposite triangle of the edge as neighbour of T
				self.triangles.insert(t, TStruct(tri_op, None, None));

				// Try to set T as neighbour of the opposite triangle
				// search the neighbour of tri_op that use edge (e1, e0)
				if let Some(tri_op) = tri_op {
					let updated_tstruct = self.triangles[&tri_op].update_with_neighbour(e0, e1, t);
					self.triangles.insert(tri_op, updated_tstruct);
				}

	            // Add triangle to a temporal list
	            new_triangles.push(t);
				
			}

			// Link the new triangles each another
			let n = new_triangles.len();
			for (i, t) in new_triangles.iter().enumerate() {
				let tstruct = self.triangles[t];
				let first_triangle = new_triangles[next_idx(i, n)];
				let second_triangle = new_triangles[prev_idx(i, n)];
				let new_tstruct = TStruct(tstruct.0, Some(first_triangle), Some(second_triangle));
				self.triangles.insert(*t, new_tstruct);
			}
		}

		pub fn export_triangles(&self) -> Vec<Triangle> {
			self.triangles.keys().filter(|t| { t.0 > 3 && t.1 > 3 && t.2 > 3 }).cloned().map(|t| { Triangle(t.0 - 4, t.1 - 4, t.2 - 4) }).collect()
		}
		
		pub fn export_voronoi_regions(&self) -> (Vec<Point>, HashMap<usize, Vec<usize>>) {
			let mut use_vertex = (0..self.coords.len()).map(|_| -> Vec<Triangle> { vec!() }).collect::<Vec<_>>();
			let mut index: HashMap<Triangle, usize> = HashMap::new();
			let mut vor_coors = vec!();
			for (tidx, t) in self.triangles.keys().enumerate() {
				let Triangle(a,b,c) = *t;
				vor_coors.push(self.circles[t].0);
				// Insert triangle, rotating it so the key is the "last" vertex 
				use_vertex[a].push(Triangle(b, c, a));
				use_vertex[b].push(Triangle(c, a, b));
				use_vertex[c].push(Triangle(a, b, c));
				// Set tidx as the index to use with this triangles
				index.insert(Triangle(b, c, a), tidx);
				index.insert(Triangle(c, a, b), tidx);
				index.insert(Triangle(a, b, c), tidx);
			}
			// init regions per coordinate dictionary
			let mut regions = HashMap::new();
			// Sort each region in a coherent order, and substitude each triangle
			// by its index

			for (i, vertex) in use_vertex[4..].iter().enumerate() {
				let mut v = vertex[0].0;
				let mut r = vec!();
				for _ in 0..vertex.len() {
					// Search the triangle beginning with vertex v
					let t = vertex.iter().find(|&t| { t.0 == v }).unwrap(); 
					r.push(t.1);											// Add the index of this triangle to region
					v = t.1;												// Choose the next vertex to search
				}
				regions.insert(i, r);										// Store region.
			}
			(vor_coors, regions)
		}
	}

	impl Triangle {
		fn circumcenter(&self, coords: &[Point]) -> (Point, f64) {

			let (a, b, c) = (coords[self.0], coords[self.1], coords[self.2]);
			/* Use coordinates relative to point `a' of the triangle. */
			let ba = b - a;
			let ca = c - a;
			// Squares of lengths of the edges incident to `a`
			let ba_length = ba.mag();
			let ca_length = ca.mag();

			// Living dangerously
			let denominator = 0.5 / (ba.x * ca.y - ba.y * ca.x);

			let xcirca = (ca.y * ba_length - ba.y * ca_length) * denominator;
			let ycirca = (ba.x * ca_length - ca.x * ba_length) * denominator;

			let a_relative_circumcenter = Point { x: xcirca, y: ycirca };

			let r_squared = a_relative_circumcenter.mag();

			(a + a_relative_circumcenter, r_squared)
		}

	}
	#[test]
	fn test_circumcenter() {
		let coords = [Point { x: -9999f64, y: -9999f64 }, Point { x: 9999f64, y: -9999f64 }, 
			Point { x: 9999f64, y: 9999f64 }, Point { x: -9999f64, y: 9999f64 }, Point { x: 13f64, y: 12f64 }];
		let t = Triangle(4,1,2);
		let (circumcenter, _radius)  = t.circumcenter(&coords);
		assert!(circumcenter.x > -10000f64);
	}
}

#[cfg(test)]
mod tests {
	use delaunay::Delaunay2D;
	use delaunay::Point;
	use delaunay::Triangle;
	use delaunay::TStruct;
    #[test]
    fn it_works() {
    	let mut delaunay = Delaunay2D::new(Point::new(0f64, 0f64), 9999f64);
    	delaunay.add_point(Point::new(13f64, 12f64));
    	for &(t, tstruct) in [
    		(Triangle(4, 0, 1), TStruct(None, Some(Triangle(4, 1, 2)), Some(Triangle(4, 3, 0)))),
    		(Triangle(4, 1, 2), TStruct(None, Some(Triangle(4, 2, 3)), Some(Triangle(4, 0, 1)))),
    		(Triangle(4, 3, 0), TStruct(None, Some(Triangle(4, 0, 1)), Some(Triangle(4, 2, 3)))),
    		(Triangle(4, 2, 3), TStruct(None, Some(Triangle(4, 3, 0)), Some(Triangle(4, 1, 2))))
    	].into_iter()
		{
			assert!(delaunay.triangles.contains_key(&t));
			let ts = delaunay.triangles[&t];
			assert_eq!(tstruct, ts);
		}

    	delaunay.add_point(Point::new(18f64, 19f64));
    	for &(t, tstruct) in [
	    	(Triangle(4, 0, 1), TStruct(None, Some(Triangle(5, 4, 1)), Some(Triangle(4, 3, 0)))),
			(Triangle(5, 2, 3), TStruct(None, Some(Triangle(5, 3, 4)), Some(Triangle(5, 1, 2)))),
			(Triangle(5, 1, 2), TStruct(None, Some(Triangle(5, 2, 3)), Some(Triangle(5, 4, 1)))),
			(Triangle(5, 3, 4), TStruct(Some(Triangle(4, 3, 0)), Some(Triangle(5, 4, 1)), Some(Triangle(5, 2, 3)))),
			(Triangle(5, 4, 1), TStruct(Some(Triangle(4, 0, 1)), Some(Triangle(5, 1, 2)), Some(Triangle(5, 3, 4)))),
			(Triangle(4, 3, 0), TStruct(None, Some(Triangle(4, 0, 1)), Some(Triangle(5, 3, 4))))
		].into_iter()
		{
			assert!(delaunay.triangles.contains_key(&t));
			let ts = delaunay.triangles[&t];
			assert_eq!(tstruct, ts);
		}

		delaunay.add_point(Point::new(21f64, 5f64));
		for &(t, tstruct) in [
			(Triangle(6, 2, 5), TStruct(Some(Triangle(5, 2, 3)), Some(Triangle(6, 5, 4)), Some(Triangle(6, 1, 2)))),
			(Triangle(5, 2, 3), TStruct(None, Some(Triangle(5, 3, 4)), Some(Triangle(6, 2, 5)))),
			(Triangle(5, 3, 4), TStruct(Some(Triangle(4, 3, 0)), Some(Triangle(6, 5, 4)), Some(Triangle(5, 2, 3)))),
			(Triangle(6, 5, 4), TStruct(Some(Triangle(5, 3, 4)), Some(Triangle(6, 4, 0)), Some(Triangle(6, 2, 5)))),
			(Triangle(6, 0, 1), TStruct(None, Some(Triangle(6, 1, 2)), Some(Triangle(6, 4, 0)))),
			(Triangle(4, 3, 0), TStruct(None, Some(Triangle(6, 4, 0)), Some(Triangle(5, 3, 4)))),
			(Triangle(6, 1, 2), TStruct(None, Some(Triangle(6, 2, 5)), Some(Triangle(6, 0, 1)))),
			(Triangle(6, 4, 0), TStruct(Some(Triangle(4, 3, 0)), Some(Triangle(6, 0, 1)), Some(Triangle(6, 5, 4)))),
		].into_iter()
		{
			assert!(delaunay.triangles.contains_key(&t));
			let ts = delaunay.triangles[&t];
			assert_eq!(tstruct, ts);
		}


		delaunay.add_point(Point::new(37f64, -3f64));

		for &(t, tstruct) in [
			(Triangle(5, 2, 3), TStruct(None, Some(Triangle(5, 3, 4)), Some(Triangle(7, 2, 5)))),
			(Triangle(7, 1, 2), TStruct(None, Some(Triangle(7, 2, 5)), Some(Triangle(7, 0, 1)))),
			(Triangle(6, 4, 0), TStruct(Some(Triangle(4, 3, 0)), Some(Triangle(7, 6, 0)), Some(Triangle(6, 5, 4)))),
			(Triangle(7, 0, 1), TStruct(None, Some(Triangle(7, 1, 2)), Some(Triangle(7, 6, 0)))),
			(Triangle(7, 6, 0), TStruct(Some(Triangle(6, 4, 0)), Some(Triangle(7, 0, 1)), Some(Triangle(7, 5, 6)))),
			(Triangle(5, 3, 4), TStruct(Some(Triangle(4, 3, 0)), Some(Triangle(6, 5, 4)), Some(Triangle(5, 2, 3)))),
			(Triangle(6, 5, 4), TStruct(Some(Triangle(5, 3, 4)), Some(Triangle(6, 4, 0)), Some(Triangle(7, 5, 6)))),
			(Triangle(7, 5, 6), TStruct(Some(Triangle(6, 5, 4)), Some(Triangle(7, 6, 0)), Some(Triangle(7, 2, 5)))),
			(Triangle(4, 3, 0), TStruct(None, Some(Triangle(6, 4, 0)), Some(Triangle(5, 3, 4)))),
			(Triangle(7, 2, 5), TStruct(Some(Triangle(5, 2, 3)), Some(Triangle(7, 5, 6)), Some(Triangle(7, 1, 2)))),
		].into_iter()
		{
			assert!(delaunay.triangles.contains_key(&t));
			let ts = delaunay.triangles[&t];
			assert_eq!(tstruct, ts);
		}
		assert_eq!(10, delaunay.triangles.len());

    }
}

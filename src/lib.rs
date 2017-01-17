mod delaunay {
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
				Point { x: center.x - radius, y: center.y + radius },
				Point { x: center.x + radius, y: center.y + radius });
			let mut triangles = HashMap::new();
			let mut circles = HashMap::new();
			let t1 = Triangle(0, 1, 3);
			let t2 = Triangle(2, 3, 1);
			triangles.insert(t1, TStruct(Some(t2), None, None));
			triangles.insert(t2, TStruct(Some(t1), None, None));
			circles.insert(t1, t1.circumcenter(&coords));
			circles.insert(t2, t2.circumcenter(&coords));
			println!("{:?}", circles);
			Delaunay2D { coords: coords, triangles: triangles, circles: circles }
		}

		pub fn dump(&self) {
			for (triangle, opposites) in self.triangles.iter() {
				println!("{:?} => {:?}", triangle, opposites);
			}
			println!("");
		}

		fn in_circle_fast(&self, tri: Triangle, p: Point) -> bool {
			let (center, radius) = self.circles[&tri];
			(center - p).mag() <= radius
		}

		fn in_circle_robust(&self, tri: Triangle, p: Point) -> bool {
			let (a, b, c) = (self.coords[tri.0] - p, self.coords[tri.1] - p, self.coords[tri.2] - p);
			let a_mag = a.mag();
			let b_mag = b.mag();
			let c_mag = c.mag();
			let det = a.x * (b.y * c_mag - b_mag * c.y)
			        + a.y * (b_mag * c.x - c_mag * b.x)
			        + a_mag * (b.x * c.y - c.x * b.y);

			det > 0f64
		}

		pub fn add_point(&mut self, p: Point) {
			let idx = self.coords.len();
			self.coords.push(p);

			let bad_triangles: HashSet<_> = self.triangles.keys().cloned().filter(|&t| self.in_circle_fast(t, p)).collect();
			println!("Bad triangles: {:?}", bad_triangles);
			
			let mut boundary: Vec<(usize, usize, Option<Triangle>)> = vec!();
			let mut t: Triangle = *bad_triangles.iter().next().unwrap();
			println!("Triangle: {:?}", t);
			let mut edge = 0;
			while true {
				println!("Edge: {}", edge);
				println!("T: {:?}", t);
				// Check if edge of triangle T is on the boundary...
	            // if opposite triangle of this edge is external to the list
				let foo = self.triangles[&t];
				println!("All opposite: {:?}", foo);
				println!("Tri_op: {:?}", foo.get(edge) );
				let tri_op = self.triangles[&t].get(edge);
				println!("boundary: {:?}", boundary);
				if tri_op.is_none() || !bad_triangles.contains(&tri_op.unwrap()) {

                	// Insert edge and external triangle into boundary list
                	println!("Adding (t.get({:?}), t.get({:?}), tri_op) = ({:?}, {:?}, {:?})", next_idx(edge, 3), prev_idx(edge, 3),
                		t.get(next_idx(edge, 3)), t.get(prev_idx(edge, 3)), tri_op);
					boundary.push((t.get(next_idx(edge, 3)), t.get(prev_idx(edge, 3)), tri_op));
					println!("new boundary: {:?}", boundary);
                	// Move to next CCW edge in this triangle
					edge = next_idx(edge, 3);
					println!("Start: {:?}; End: {:?}", boundary[0].0, boundary[boundary.len() - 1].1);
					if boundary[0].0 == boundary[boundary.len() - 1].1 {
						break;
					}
				} else if let Some(tri_op) = tri_op {
					// Move to next CCW edge in opposite triangle
					let ccw_op = self.triangles[&tri_op].get_ccw_op(t);
					//edge = (self.triangles[tri_op].index(T) + 1) % 3
					println!("CCW_OP: {:?} of {:?} for {:?}", ccw_op, self.triangles.get(&tri_op).unwrap(), t);
					edge = next_idx(ccw_op, 3);
					println!("New edge: {:?}", edge);
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
					if tri_op == Triangle(4, 0, 1) {
						println!("Oi! {} {} {:?} -> {:?}", e0, e1, t, updated_tstruct);
					}
					self.triangles.insert(tri_op, updated_tstruct);
				}

	            // Add triangle to a temporal list
	            new_triangles.push(t);
				
			}

			// Link the new triangles each another
			let n = new_triangles.len();
			println!("new_triangles.len: {:?}", n);
			for (i, t) in new_triangles.iter().enumerate() {
				let tstruct = self.triangles[t];
				let first_triangle = new_triangles[next_idx(i, n)];
				let second_triangle = new_triangles[prev_idx(i, n)];
				let new_tstruct = TStruct(tstruct.0, Some(first_triangle), Some(second_triangle));
				self.triangles.insert(*t, new_tstruct);
			}
			println!("");
			println!("");
			println!("");
		}

		pub fn export_triangles(&self) -> Vec<Triangle> {
			self.triangles.keys().filter(|t| { t.0 > 3 && t.1 > 3 && t.2 > 3 }).cloned().map(|t| { Triangle(t.0 - 3, t.1 - 3, t.2 - 3) }).collect()
		}
		//     def exportTriangles(self):
//         """Export the current list of Delaunay triangles
//         """
//         # Filter out triangles with any vertex in the extended BBox
//         return [(a-4, b-4, c-4)
//                 for (a, b, c) in self.triangles if a > 3 and b > 3 and c > 3]

	}

	impl Triangle {
		fn circumcenter(&self, coords: &[Point]) -> (Point, f64) {
			let (a, b, c) = (coords[self.0], coords[self.1], coords[self.2]);
			let d = 2f64 * (a.y * c.x + b.y * a.x - b.y * c.x - a.y * b.x - c.y * a.x + c.y * b.x);
			println!("D: {:?}", d);
			let p_0 =
				( b.y * a.x * a.x
				- c.y * a.x * a.x
				- b.y * b.y * a.y
				+ c.y * c.y * a.y
				+ b.x * b.x * c.y
				+ a.y * a.y * b.y
				+ c.x * c.x * a.y
				- c.y * c.y * b.y
				- c.x * c.x * b.y 
				- b.x * b.x * a.y
				+ b.y * b.y * c.y
				- a.y * a.y * c.y )
				 / d;
			let p_1 =
				( a.x * a.x * c.x
				+ a.y * a.y * c.x
				+ b.x * b.x * a.x
				- b.x * b.x * c.x
				+ b.y * b.y * a.x
				- b.y * b.y * c.x
				- a.x * a.x * b.x
				- a.y * a.y * b.x
				- c.x * c.x * a.x
				+ c.x * c.x * b.x
				- c.y * c.y * a.x
				+ c.y * c.y * b.x) 
				 / d;

			let r_squared = (a.x - p_0) * (a.x - p_0) + (a.y - p_1) * (a.y - p_1);

			(Point { x: p_0, y: p_1 }, r_squared)
		}

	}
	#[test]
	fn test_circumcenter() {
		let coords = [Point { x: -9999f64, y: -9999f64 }, Point { x: 9999f64, y: -9999f64 }, 
			Point { x: -9999f64, y: 9999f64 }, Point { x: 9999f64, y: 9999f64 }, Point { x: 13f64, y: 12f64 }];
		let t = Triangle(4,1,2);
		let (circumcenter, radius)  = t.circumcenter(&coords);
		println!("{:?}", circumcenter);
		assert!(circumcenter.x > -10000f64);
	}
}

//     def exportTriangles(self):
//         """Export the current list of Delaunay triangles
//         """
//         # Filter out triangles with any vertex in the extended BBox
//         return [(a-4, b-4, c-4)
//                 for (a, b, c) in self.triangles if a > 3 and b > 3 and c > 3]

//     def exportCircles(self):
//         """Export the circumcircles as a list of (center, radius)
//         """
//         # Remember to compute circumcircles if not done before
//         # for t in self.triangles:
//         #     self.circles[t] = self.Circumcenter(t)

//         # Filter out triangles with any vertex in the extended BBox
//         # Do sqrt of radius before of return
//         return [(self.circles[(a, b, c)][0], sqrt(self.circles[(a, b, c)][1]))
//                 for (a, b, c) in self.triangles if a > 3 and b > 3 and c > 3]

//     def exportDT(self):
//         """Export the current set of Delaunay coordinates and triangles.
//         """
//         # Filter out coordinates in the extended BBox
//         coord = self.coords[4:]

//         # Filter out triangles with any vertex in the extended BBox
//         tris = [(a-4, b-4, c-4)
//                 for (a, b, c) in self.triangles if a > 3 and b > 3 and c > 3]
//         return coord, tris

//     def exportExtendedDT(self):
//         """Export the Extended Delaunay Triangulation (with the frame vertex).
//         """
//         return self.coords, list(self.triangles)
        
//     def exportVoronoiRegions(self):
//         """Export coordinates and regions of Voronoi diagram as indexed data.
//         """
//         # Remember to compute circumcircles if not done before
//         # for t in self.triangles:
//         #     self.circles[t] = self.Circumcenter(t)
//         useVertex = {i:[] for i in range(len(self.coords))}
//         vor_coors = []
//         index={}
//         # Build a list of coordinates and a index per triangle/region
//         for tidx, (a, b, c) in enumerate(self.triangles):
//             vor_coors.append(self.circles[(a,b,c)][0])
//             # Insert triangle, rotating it so the key is the "last" vertex 
//             useVertex[a]+=[(b, c, a)]
//             useVertex[b]+=[(c, a, b)]
//             useVertex[c]+=[(a, b, c)]
//             # Set tidx as the index to use with this triangles
//             index[(a, b, c)] = tidx;
//             index[(c, a, b)] = tidx;
//             index[(b, c, a)] = tidx;
            
//         # init regions per coordinate dictionary
//         regions = {}
//         # Sort each region in a coherent order, and substitude each triangle
//         # by its index
//         for i in range (4, len(self.coords)):
//             v = useVertex[i][0][0]  # Get a vertex of a triangle
//             r=[]
//             for _ in range(len(useVertex[i])):
//                 # Search the triangle beginning with vertex v
//                 t = [t for t in useVertex[i] if t[0] == v][0]
//                 r.append(index[t])  # Add the index of this triangle to region
//                 v = t[1]            # Choose the next vertex to search
//             regions[i-4]=r          # Store region.
            
//         return vor_coors, regions
// }

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
    	println!("Added first p[oint");
    	// {(4, 0, 1): [None, (4, 1, 2), (4, 3, 0)], (4, 1, 2): [None, (4, 2, 3), (4, 0, 1)], (4, 3, 0): [None, (4, 0, 1), (4, 2, 3)], (4, 2, 3): [None, (4, 3, 0), (4, 1, 2)]}
		{
			let t = Triangle(4, 0, 1);
			assert!(delaunay.triangles.contains_key(&t));
			let ts = delaunay.triangles[&t];
			assert_eq!(TStruct(None, Some(Triangle(4, 1, 2)), Some(Triangle(4, 3, 0))), ts);
		}
		{
			let t = Triangle(4, 1, 2);
			assert!(delaunay.triangles.contains_key(&t));
			let ts = delaunay.triangles[&t];
			assert_eq!(TStruct(None, Some(Triangle(4, 2, 3)), Some(Triangle(4, 0, 1))), ts);
		}
		{
			let t = Triangle(4, 3, 0);
			assert!(delaunay.triangles.contains_key(&t));
			let ts = delaunay.triangles[&t];
			assert_eq!(TStruct(None, Some(Triangle(4, 0, 1)), Some(Triangle(4, 2, 3))), ts);
		}
		{
			let t = Triangle(4, 2, 3);
			assert!(delaunay.triangles.contains_key(&t));
			let ts = delaunay.triangles[&t];
			assert_eq!(TStruct(None, Some(Triangle(4, 3, 0)), Some(Triangle(4, 1, 2))), ts);
		}

		println!("{:?}", delaunay.coords);

    	delaunay.add_point(Point::new(18f64, 19f64));
    	println!("Added second p[oint");
    	{
			let t = Triangle(4, 0, 1);
			assert!(delaunay.triangles.contains_key(&t));
			let ts = delaunay.triangles[&t];
			assert_eq!(TStruct(None, Some(Triangle(5, 4, 1)), Some(Triangle(4, 3, 0))), ts);
		}
		{
			let t = Triangle(5, 2, 3);
			assert!(delaunay.triangles.contains_key(&t));
			let ts = delaunay.triangles[&t];
			assert_eq!(TStruct(None, Some(Triangle(5, 3, 4)), Some(Triangle(5, 1, 2))), ts);
		}
		{
			let t = Triangle(5, 1, 2);
			assert!(delaunay.triangles.contains_key(&t));
			let ts = delaunay.triangles[&t];
			assert_eq!(TStruct(None, Some(Triangle(5, 2, 3)), Some(Triangle(5, 4, 1))), ts);
		}
		{
			let t = Triangle(5, 3, 4);
			assert!(delaunay.triangles.contains_key(&t));
			let ts = delaunay.triangles[&t];
			assert_eq!(TStruct(Some(Triangle(4, 3, 0)), Some(Triangle(5, 4, 1)), Some(Triangle(5, 2, 3))), ts);
		}
		{
			let t = Triangle(5, 4, 1);
			assert!(delaunay.triangles.contains_key(&t));
			let ts = delaunay.triangles[&t];
			assert_eq!(TStruct(Some(Triangle(4, 0, 1)), Some(Triangle(5, 1, 2)), Some(Triangle(5, 3, 4))), ts);
		}
		{
			let t = Triangle(4, 3, 0);
			assert!(delaunay.triangles.contains_key(&t));
			let ts = delaunay.triangles[&t];
			assert_eq!(TStruct(None, Some(Triangle(4, 0, 1)), Some(Triangle(5, 3, 4))), ts);
		}


		delaunay.add_point(Point::new(21f64, 5f64));
    	println!("Added third p[oint");
		{
			let t = Triangle(6, 2, 5);
			assert!(delaunay.triangles.contains_key(&t));
			let ts = delaunay.triangles[&t];
			assert_eq!(TStruct(Some(Triangle(5, 2, 3)), Some(Triangle(6, 5, 4)), Some(Triangle(6, 1, 2))), ts);
		}
		{
			let t = Triangle(5, 2, 3);
			assert!(delaunay.triangles.contains_key(&t));
			let ts = delaunay.triangles[&t];
			assert_eq!(TStruct(None, Some(Triangle(5, 3, 4)), Some(Triangle(6, 2, 5))), ts);
		}
		{
			let t = Triangle(5, 3, 4);
			assert!(delaunay.triangles.contains_key(&t));
			let ts = delaunay.triangles[&t];
			assert_eq!(TStruct(Some(Triangle(4, 3, 0)), Some(Triangle(6, 5, 4)), Some(Triangle(5, 2, 3))), ts);
		}
		{
			let t = Triangle(6, 5, 4);
			assert!(delaunay.triangles.contains_key(&t));
			let ts = delaunay.triangles[&t];
			assert_eq!(TStruct(Some(Triangle(5, 3, 4)), Some(Triangle(6, 4, 0)), Some(Triangle(6, 2, 5))), ts);
		}
		{
			let t = Triangle(6, 0, 1);
			assert!(delaunay.triangles.contains_key(&t));
			let ts = delaunay.triangles[&t];
			assert_eq!(TStruct(None, Some(Triangle(6, 1, 2)), Some(Triangle(6, 4, 0))), ts);
		}
		{
			let t = Triangle(4, 3, 0);
			assert!(delaunay.triangles.contains_key(&t));
			let ts = delaunay.triangles[&t];
			assert_eq!(TStruct(None, Some(Triangle(6, 4, 0)), Some(Triangle(5, 3, 4))), ts);
		}
		{
			let t = Triangle(6, 1, 2);
			assert!(delaunay.triangles.contains_key(&t));
			let ts = delaunay.triangles[&t];
			assert_eq!(TStruct(None, Some(Triangle(6, 2, 5)), Some(Triangle(6, 0, 1))), ts);
		}
		{
			let t = Triangle(6, 4, 0);
			assert!(delaunay.triangles.contains_key(&t));
			let ts = delaunay.triangles[&t];
			assert_eq!(TStruct(Some(Triangle(4, 3, 0)), Some(Triangle(6, 0, 1)), Some(Triangle(6, 5, 4))), ts);
		}

		delaunay.add_point(Point::new(37f64, -3f64));

		{
			let t = Triangle(5, 2, 3);
			assert!(delaunay.triangles.contains_key(&t));
			let ts = delaunay.triangles[&t];
			assert_eq!(TStruct(None, Some(Triangle(5, 3, 4)), Some(Triangle(7, 2, 5))), ts);
		}
		{
			let t = Triangle(7, 1, 2);
			assert!(delaunay.triangles.contains_key(&t));
			let ts = delaunay.triangles[&t];
			assert_eq!(TStruct(None, Some(Triangle(7, 2, 5)), Some(Triangle(7, 0, 1))), ts);
		}
		{
			let t = Triangle(6, 4, 0);
			assert!(delaunay.triangles.contains_key(&t));
			let ts = delaunay.triangles[&t];
			assert_eq!(TStruct(Some(Triangle(4, 3, 0)), Some(Triangle(7, 6, 0)), Some(Triangle(6, 5, 4))), ts);
		}
		{
			let t = Triangle(7, 0, 1);
			assert!(delaunay.triangles.contains_key(&t));
			let ts = delaunay.triangles[&t];
			assert_eq!(TStruct(None, Some(Triangle(7, 1, 2)), Some(Triangle(7, 6, 0))), ts);
		}
		{
			let t = Triangle(7, 6, 0);
			assert!(delaunay.triangles.contains_key(&t));
			let ts = delaunay.triangles[&t];
			assert_eq!(TStruct(Some(Triangle(6, 4, 0)), Some(Triangle(7, 0, 1)), Some(Triangle(7, 5, 6))), ts);
		}
		{
			let t = Triangle(5, 3, 4);
			assert!(delaunay.triangles.contains_key(&t));
			let ts = delaunay.triangles[&t];
			assert_eq!(TStruct(Some(Triangle(4, 3, 0)), Some(Triangle(6, 5, 4)), Some(Triangle(5, 2, 3))), ts);
		}
		{
			let t = Triangle(6, 5, 4);
			assert!(delaunay.triangles.contains_key(&t));
			let ts = delaunay.triangles[&t];
			assert_eq!(TStruct(Some(Triangle(5, 3, 4)), Some(Triangle(6, 4, 0)), Some(Triangle(7, 5, 6))), ts);
		}
		{
			let t = Triangle(7, 5, 6);
			assert!(delaunay.triangles.contains_key(&t));
			let ts = delaunay.triangles[&t];
			assert_eq!(TStruct(Some(Triangle(6, 5, 4)), Some(Triangle(7, 6, 0)), Some(Triangle(7, 2, 5))), ts);
		}
		{
			let t = Triangle(4, 3, 0);
			assert!(delaunay.triangles.contains_key(&t));
			let ts = delaunay.triangles[&t];
			assert_eq!(TStruct(None, Some(Triangle(6, 4, 0)), Some(Triangle(5, 3, 4))), ts);
		}
		{
			let t = Triangle(7, 2, 5);
			assert!(delaunay.triangles.contains_key(&t));
			let ts = delaunay.triangles[&t];
			assert_eq!(TStruct(Some(Triangle(5, 2, 3)), Some(Triangle(7, 5, 6)), Some(Triangle(7, 1, 2))), ts);
		}



    	// delaunay.dump();
    	// delaunay.add_point(Point::new(18f64, 19f64));
    	// delaunay.dump();
    	// 
    	// delaunay.dump();
    	// 
    	// delaunay.dump();
    	// println!("{:?}", delaunay.export_triangles());
    	// assert!(false); 
    }
}

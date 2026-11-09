//! The tiny-tier scenarios (PLAN milestone 2, SPEC §11). Each is a `.osm.pbf`, an Overture
//! release and a config directory, plus the facts the conformance suite checks on top of
//! its general checks.

use crate::fixtures::build::{Builder, Frame, Via};
use crate::fixtures::overture::{Category, Redirect};
use crate::fixtures::{Check, Scenario, Variant};

/// Walking speed of the ACTIVE profile, m/s: a length of `t / WALK` metres is t seconds.
const WALK: f64 = 5.0 / 3.6;
/// Car speed on a residential way without maxspeed, m/s.
const RESIDENTIAL: f64 = 25.0 / 3.6;

pub fn all() -> Vec<Scenario> {
    vec![toy(), town(), cap(), viaway(), deadend(), fragment(), cells()]
}

fn build_toml(target: u64, budget: u64, pack: u64, order: Option<&[i64]>) -> String {
    let mut s = format!(
        "[build]\nshard_target_bytes = {target}\nshard_max_bytes = {}\nmemory_budget_bytes = {budget}\npack_max_nodes = {pack}\n",
        2 * target
    );
    if let Some(o) = order {
        let ids: Vec<String> = o.iter().map(|x| x.to_string()).collect();
        s.push_str(&format!("\n[order]\nactive = [{}]\n", ids.join(", ")));
    }
    s
}

fn taxonomy(b: &mut Builder) {
    let cat = |h: &[&str], is_basic: bool, basic: &str| Category {
        hierarchy: h.iter().map(|s| s.to_string()).collect(),
        is_basic,
        basic: Some(basic.to_string()),
    };
    b.release.categories = vec![
        cat(&["eat_and_drink"], true, "eat_and_drink"),
        cat(&["eat_and_drink", "restaurant"], true, "restaurant"),
        cat(&["eat_and_drink", "restaurant", "pizza_restaurant"], false, "restaurant"),
        cat(&["eat_and_drink", "cafe"], true, "cafe"),
        cat(&["health_care"], true, "health_care"),
        cat(&["health_care", "pharmacy"], true, "pharmacy"),
        cat(&["travel"], true, "travel"),
        cat(&["travel", "parking"], true, "parking"),
    ];
    b.release.redirects = vec![
        Redirect { old: "pizzeria".into(), new: Some("pizza_restaurant".into()) },
        Redirect { old: "drugstore".into(), new: Some("pharmacy".into()) },
        Redirect { old: "gone_category".into(), new: None },
    ];
}

/// A grid of nodes, `n[i][j]` at (x0 + i·s, y0 + j·s).
fn grid(b: &mut Builder, x0: f64, y0: f64, nx: usize, ny: usize, s: f64) -> Vec<Vec<i64>> {
    (0..nx).map(|i| (0..ny).map(|j| b.node(x0 + i as f64 * s, y0 + j as f64 * s)).collect()).collect()
}

/// The 3×3 toy town of the research (SPEC §11): eleven two-way footways and one one-way, with
/// lengths that make their walking times 1–5 s, and the hand-made order. 17 links, 5 levels.
fn toy() -> Scenario {
    let mut b = Builder::new(Frame::new(50.0, 3.0), 1);
    const TWO_WAY: [(usize, usize, u32); 11] =
        [(0, 1, 2), (3, 4, 4), (4, 5, 1), (6, 7, 5), (7, 8, 2), (0, 3, 1), (3, 6, 4), (1, 4, 2), (4, 7, 3), (2, 5, 3), (5, 8, 1)];
    // Nodes 1 m apart: node k at column k % 3, row k / 3.
    let pos = |k: usize| ((k % 3) as f64, (k / 3) as f64);
    let nodes: Vec<i64> = (0..9).map(|k| b.node(pos(k).0, pos(k).1)).collect();
    // One bend in the middle of each edge gives it the length of its time.
    let edge = |b: &mut Builder, u: usize, v: usize, t: u32, oneway: bool| {
        let ((xu, yu), (xv, yv)) = (pos(u), pos(v));
        let l = t as f64 * WALK;
        let h = ((l / 2.0).powi(2) - 0.25).sqrt();
        let (mx, my) = ((xu + xv) / 2.0 + h * (yv - yu), (yu + yv) / 2.0 + h * (xu - xv));
        let mid = b.node(mx, my);
        let mut t = vec![("highway", "footway"), ("bicycle", "no")];
        if oneway {
            t.push(("oneway:foot", "yes"));
        }
        b.way(&[nodes[u], mid, nodes[v]], &t);
    };
    for &(u, v, t) in &TWO_WAY {
        edge(&mut b, u, v, t, false);
    }
    edge(&mut b, 1, 2, 4, true);
    taxonomy(&mut b);
    let ids: Vec<String> = (0..9).map(|k| b.address(pos(k).0, pos(k).1)).collect();
    // Grid times from the research graph (Dijkstra over the same arcs).
    let mut arcs: Vec<(usize, usize, u32)> = TWO_WAY.iter().flat_map(|&(u, v, t)| [(u, v, t), (v, u, t)]).collect();
    arcs.push((1, 2, 4));
    let times: Vec<Vec<u16>> = (0..9).map(|s| dijkstra(9, &arcs, s).into_iter().map(|d| d as u16).collect()).collect();
    assert_eq!(times[0], [0, 2, 6, 1, 4, 5, 5, 7, 6]);
    let order: Vec<i64> = [0, 6, 3, 2, 8, 5, 1, 7, 4].iter().map(|&k| nodes[k]).collect();
    Scenario {
        name: "toy",
        osm: b.osm,
        release: b.release,
        extent: None,
        variants: vec![Variant { name: "default", build_toml: build_toml(1 << 20, 1 << 30, 0, Some(&order)) }],
        checks: vec![
            Check::Hierarchy { family: "active", component: 0, links: 17, levels: 5 },
            Check::Grid { mode: "walk", ids: ids.clone(), times },
            Check::Reason { id: ids[0].clone(), mode: "bike", reason: "NoEligibleRoad" },
        ],
    }
}

fn dijkstra(n: usize, arcs: &[(usize, usize, u32)], s: usize) -> Vec<u32> {
    let mut d = vec![u32::MAX; n];
    let mut done = vec![false; n];
    d[s] = 0;
    for _ in 0..n {
        let Some(u) = (0..n).filter(|&v| !done[v] && d[v] != u32::MAX).min_by_key(|&v| d[v]) else { break };
        done[u] = true;
        for &(a, b, t) in arcs {
            if a == u && d[u] + t < d[b] {
                d[b] = d[u] + t;
            }
        }
    }
    d
}

/// A town with most of what SPEC §5, §6 and §9 distinguish: one-ways, signals, node-via
/// restrictions, an island, a car trap and a walk trap, a footbridge for walkers only, a
/// motorway alone, same-piece pairs, ties, and places with every rule of SPEC §3.
fn town() -> Scenario {
    let mut b = Builder::new(Frame::new(50.6, 3.06), 2);
    let res = [("highway", "residential")];
    // Main grid: 6 × 6 nodes, 100 m apart.
    let g = grid(&mut b, 0.0, 0.0, 6, 6, 100.0);
    let row = |j: usize, r: std::ops::RangeInclusive<usize>| -> Vec<i64> { r.map(|i| g[i][j]).collect() };
    let col = |i: usize, r: std::ops::RangeInclusive<usize>| -> Vec<i64> { r.map(|j| g[i][j]).collect() };
    for j in [0, 1, 4, 5] {
        b.way(&row(j, 0..=5), &res);
    }
    b.way(&row(2, 0..=5), &[("highway", "residential"), ("oneway", "yes")]);
    let row3_west = b.way(&row(3, 0..=3), &res);
    b.way(&row(3, 3..=5), &res);
    b.way(&col(0, 0..=5), &res);
    let col1_south = b.way(&col(1, 0..=1), &[("highway", "tertiary"), ("maxspeed", "30")]);
    let col1_north = b.way(&col(1, 1..=5), &[("highway", "tertiary"), ("maxspeed", "30")]);
    b.way(&col(2, 0..=5), &res);
    b.way(&col(3, 0..=3), &res);
    let col3_north = b.way(&col(3, 3..=5), &res);
    b.way(&col(4, 0..=5), &[("highway", "residential"), ("oneway", "-1")]);
    b.way(&col(5, 0..=5), &[("highway", "service")]);
    b.tag_node(g[2][3], &[("highway", "traffic_signals")]);
    b.tag_node(g[4][4], &[("highway", "stop")]);
    b.restriction("no_left_turn", row3_west, Via::Node(g[3][3]), col3_north);
    b.restriction("only_straight_on", col1_south, Via::Node(g[1][1]), col1_north);

    // East bank, joined to the main grid by a footbridge only.
    let e = grid(&mut b, 700.0, 100.0, 3, 3, 100.0);
    for j in 0..3 {
        b.way(&[e[0][j], e[1][j], e[2][j]], &res);
    }
    for (i, ei) in e.iter().enumerate() {
        b.way(ei, if i == 1 { &[("highway", "living_street")][..] } else { &res[..] });
    }
    b.way(&[g[5][2], e[0][1]], &[("highway", "footway"), ("bridge", "yes"), ("bicycle", "no")]);

    // Island, 3 km away: a 3 × 3 grid.
    let isl = grid(&mut b, 3000.0, 3000.0, 3, 3, 100.0);
    for j in 0..3 {
        b.way(&[isl[0][j], isl[1][j], isl[2][j]], &res);
    }
    for c in &isl {
        b.way(c, &res);
    }
    // Car trap: a one-way road 3 km north of the island to a pocket of four two-way streets.
    let p0 = b.node(3200.0, 6200.0);
    let p1 = b.node(3300.0, 6200.0);
    let p2 = b.node(3300.0, 6300.0);
    let p3 = b.node(3200.0, 6300.0);
    let mid = b.node(3200.0, 4600.0);
    b.way(&[isl[2][2], mid, p0], &[("highway", "residential"), ("oneway", "yes")]);
    b.way(&[p0, p1, p2, p3, p0], &res);
    // Walk trap: a one-way footway west of the main grid to a pocket of footways.
    let w0 = b.node(-1000.0, 500.0);
    let w1 = b.node(-1100.0, 500.0);
    let w2 = b.node(-1100.0, 600.0);
    let w3 = b.node(-1000.0, 600.0);
    let foot = [("highway", "footway"), ("bicycle", "no")];
    b.way(&[g[0][5], w0], &[("highway", "footway"), ("bicycle", "no"), ("oneway:foot", "yes")]);
    b.way(&[w0, w1, w2, w3, w0], &foot);
    // A motorway alone, 8 km east.
    let m: Vec<i64> = (0..4).map(|k| b.node(8000.0, k as f64 * 700.0)).collect();
    b.way(&m, &[("highway", "motorway")]);

    taxonomy(&mut b);
    // Addresses: one beside every piece of the main grid.
    for i in 0..6 {
        for j in 0..6 {
            let (x, y) = (i as f64 * 100.0, j as f64 * 100.0);
            if i < 5 {
                b.address(x + 30.0, y + 12.0);
            }
            if j < 5 {
                b.address(x + 12.0, y + 60.0);
            }
        }
    }
    // Same-piece pairs: equal offsets on a two-way piece; both orders on a one-way piece.
    let same_a = b.address(55.0, 8.0);
    let same_b = b.address(55.0, -8.0);
    let one_a = b.address(130.0, 210.0);
    let one_b = b.address(170.0, 190.0);
    let at_node = b.address(200.0, 200.0);
    let tie_src = b.address(200.0, 0.0);
    let east = b.address(820.0, 210.0);
    let island = b.address(3050.0, 3110.0);
    let car_trap = b.address(3250.0, 6250.0);
    let walk_trap = b.address(-1050.0, 550.0);
    let motorway = b.address(8030.0, 1000.0);
    let nowhere = b.address(20000.0, 20000.0);
    let never = b.uuid();

    // Places.
    let cafe_a = b.place(100.0, 0.0, Some("cafe"), Some("cafe"), "Cafe du Nord");
    let cafe_b = b.place(300.0, 0.0, Some("cafe"), Some("cafe"), "Cafe du Sud");
    let paul = b.place(250.0, 110.0, Some("restaurant"), Some("restaurant"), "Chez Paul");
    let paul_dup = b.place(255.0, 112.0, Some("restaurant"), Some("restaurant"), "Chez Paul SARL");
    b.last_place().confidence = Some(0.8);
    b.place(420.0, 330.0, Some("restaurant"), Some("restaurant"), "Le Bistro");
    b.place(130.0, 420.0, Some("pizza_restaurant"), Some("restaurant"), "Pizza Roma");
    b.last_place().alternates = vec!["cafe".into()];
    let pharmacy = b.place(330.0, 230.0, Some("pharmacy"), Some("pharmacy"), "Pharmacie");
    b.last_place().confidence = Some(0.55);
    let low = b.place(440.0, 120.0, Some("restaurant"), Some("restaurant"), "Doubtful");
    b.last_place().confidence = Some(0.55);
    let closed = b.place(60.0, 330.0, Some("cafe"), Some("cafe"), "Closed Cafe");
    b.last_place().operating_status = Some("permanently_closed".into());
    let bare = b.place(210.0, 470.0, None, None, "Somewhere");
    // Retired keys, redirected (SPEC §6): primary `drugstore` is a pharmacy, alternate
    // `pizzeria` a pizza restaurant.
    let drugstore = b.place(170.0, 280.0, Some("drugstore"), Some("pharmacy"), "Droguerie");
    b.last_place().alternates = vec!["pizzeria".into()];
    b.place(370.0, 480.0, Some("space_elevator"), None, "Elevator");
    b.place(20.0, 220.0, Some("parking"), Some("parking"), "Parking West");
    b.place(480.0, 380.0, Some("parking"), Some("parking"), "Parking East");
    b.place(780.0, 280.0, Some("restaurant"), Some("restaurant"), "Rive Droite");
    b.place(3150.0, 3090.0, Some("cafe"), Some("cafe"), "Island Cafe");

    let r = |id: &String, mode: &'static str, reason: &'static str| Check::Reason { id: id.clone(), mode, reason };
    Scenario {
        name: "town",
        osm: b.osm,
        release: b.release,
        extent: None,
        variants: vec![Variant { name: "default", build_toml: build_toml(16 << 10, 1 << 30, 8, None) }],
        checks: vec![
            r(&motorway, "car", "NoEligibleRoad"),
            r(&motorway, "bike", "NoRoad"),
            r(&motorway, "walk", "NoRoad"),
            r(&car_trap, "car", "TrappedComponent"),
            r(&walk_trap, "walk", "TrappedComponent"),
            r(&walk_trap, "bike", "NoEligibleRoad"),
            r(&low, "car", "FilteredPlace"),
            r(&closed, "walk", "FilteredPlace"),
            Check::Served { id: car_trap.clone(), mode: "walk" },
            Check::Served { id: walk_trap.clone(), mode: "car" },
            Check::Served { id: pharmacy, mode: "walk" },
            Check::Served { id: bare, mode: "walk" },
            Check::Alias { alias: paul_dup, kept: paul },
            Check::Unknown { id: nowhere },
            Check::Unknown { id: never },
            Check::DifferentComponent { mode: "car", a: tie_src.clone(), b: east.clone() },
            Check::DifferentComponent { mode: "car", a: tie_src.clone(), b: island.clone() },
            Check::DifferentComponent { mode: "walk", a: tie_src.clone(), b: island },
            Check::Beyond { mode: "bike", a: tie_src.clone(), b: east.clone() },
            Check::Reachable { mode: "walk", a: tie_src.clone(), b: east },
            Check::Tie { category: "cafe", id: tie_src, mode: "walk", place: cafe_a.min(cafe_b) },
            Check::Matches { category: "pharmacy", place: drugstore.clone() },
            Check::Matches { category: "pizza_restaurant", place: drugstore },
            Check::SamePiece { mode: "car", a: same_a.clone(), b: same_b.clone() },
            Check::SamePiece { mode: "walk", a: same_a, b: same_b },
            Check::SamePiece { mode: "car", a: one_a.clone(), b: one_b.clone() },
            Check::SamePiece { mode: "bike", a: one_a, b: one_b },
            Check::AtNode { id: at_node, mode: "walk" },
            Check::Trapped { family: "car", at_least: 2 },
            Check::Trapped { family: "active", at_least: 1 },
        ],
    }
}

/// Routes just under and just over the 65,534 s cap (SPEC §7, §11): car ways along the
/// equator, each one piece, with car times 30000, 35525, 20 and 20 s. On foot the first
/// two take 150,000 and 177,625 s, so they are cut into parts (SPEC §7, long pieces), and
/// IDs on them stay exact below the cap.
fn cap() -> Scenario {
    let mut b = Builder::new(Frame::new(0.0, 10.0), 3);
    let times = [30000.0, 35525.0, 20.0, 20.0];
    let mut x = 0.0;
    let mut xs = vec![0.0];
    for t in times {
        x += t * RESIDENTIAL;
        xs.push(x);
    }
    let nodes: Vec<i64> = xs.iter().map(|&x| b.node(x, 0.0)).collect();
    for w in nodes.windows(2) {
        b.way(w, &[("highway", "residential")]);
    }
    taxonomy(&mut b);
    let ids: Vec<String> = xs.iter().map(|&x| b.address(x, 0.0)).collect();
    // In the middle of the third way: 65525 + 10 = 65535 s from the start, the cap itself.
    let at_cap = b.address((xs[2] + xs[3]) / 2.0, 0.0);
    // On the first way, 60,000, 65,000 and 65,600 s on foot from its start.
    let walk = |t: f64| t * 5.0 / 3.6;
    let on_foot: Vec<String> = [60000.0, 65000.0, 65600.0].iter().map(|&t| b.address(walk(t), 0.0)).collect();
    let cap = |a: &String, c: &String, under: bool| Check::Cap { mode: "car", a: a.clone(), b: c.clone(), under };
    Scenario {
        name: "cap",
        osm: b.osm,
        release: b.release,
        extent: None,
        variants: vec![Variant { name: "default", build_toml: build_toml(1 << 20, 1 << 30, 0, None) }],
        checks: vec![
            cap(&ids[0], &ids[2], true),
            cap(&ids[2], &ids[0], true),
            cap(&ids[0], &ids[3], false),
            cap(&ids[3], &ids[0], false),
            cap(&ids[0], &at_cap, false),
            cap(&ids[1], &ids[4], true),
            Check::Beyond { mode: "walk", a: ids[0].clone(), b: ids[1].clone() },
            Check::Cap { mode: "walk", a: ids[0].clone(), b: on_foot[0].clone(), under: true },
            Check::Cap { mode: "walk", a: on_foot[0].clone(), b: ids[0].clone(), under: true },
            Check::Cap { mode: "walk", a: ids[0].clone(), b: on_foot[1].clone(), under: true },
            Check::Cap { mode: "walk", a: on_foot[1].clone(), b: ids[0].clone(), under: true },
            Check::Cap { mode: "walk", a: ids[0].clone(), b: on_foot[2].clone(), under: false },
            Check::Cap { mode: "walk", a: on_foot[0].clone(), b: on_foot[2].clone(), under: true },
        ],
    }
}

/// Via-way restrictions (SPEC §4, §11): `no_*` and `only_*` with one and two via ways, on a
/// 4 × 4 grid whose every piece is its own way, built with an extent polygon (SPEC §3).
fn viaway() -> Scenario {
    let mut b = Builder::new(Frame::new(48.0, 2.0), 4);
    let g = grid(&mut b, 0.0, 0.0, 4, 4, 100.0);
    let res = [("highway", "residential")];
    let mut h = vec![vec![0i64; 4]; 3];
    let mut v = vec![vec![0i64; 3]; 4];
    for j in 0..4 {
        for i in 0..3 {
            h[i][j] = b.way(&[g[i][j], g[i + 1][j]], &res);
        }
    }
    for (i, vi) in v.iter_mut().enumerate() {
        for (j, w) in vi.iter_mut().enumerate() {
            *w = b.way(&[g[i][j], g[i][j + 1]], &res);
        }
    }
    // A: no u-turn (1,1) east, north to (1,2), back west.
    b.restriction("no_u_turn", h[0][1], Via::Ways(vec![v[1][1]]), h[0][2]);
    // B: only straight on, north along column 2.
    b.restriction("only_straight_on", v[2][0], Via::Ways(vec![v[2][1]]), v[2][2]);
    // C: north along column 3, west along row 2, then no right turn (north) at (2, 2).
    b.restriction("no_right_turn", v[3][0], Via::Ways(vec![v[3][1], h[2][2]]), v[2][2]);
    // D: only right turn: east along row 0, via (1,0)-(2,0) and (2,0)-(2,1), then east.
    b.restriction("only_right_turn", h[0][0], Via::Ways(vec![v[2][0], h[1][0]]), h[2][1]);
    taxonomy(&mut b);
    let mut on_via = Vec::new();
    for i in 0..4 {
        for j in 0..4 {
            let (x, y) = (i as f64 * 100.0, j as f64 * 100.0);
            if i < 3 {
                b.address(x + 40.0, y + 10.0);
            }
            if j < 3 {
                let id = b.address(x + 10.0, y + 40.0);
                if (i, j) == (1, 1) || (i, j) == (2, 1) || (i, j) == (3, 1) || (i, j) == (2, 0) {
                    on_via.push(id);
                }
            }
        }
    }
    b.place(150.0, 320.0, Some("cafe"), Some("cafe"), "Corner Cafe");
    let outside = b.address(900.0, 900.0);
    let no_road = b.address(-450.0, -450.0);
    let extent = "viaway\n1\n".to_string()
        + &[(-500.0, -500.0), (500.0, -500.0), (500.0, 500.0), (-500.0, 500.0)]
            .iter()
            .map(|&(x, y)| {
                let (lat, lon) = b.frame.ll(x, y);
                format!("   {lon:.7}   {lat:.7}\n")
            })
            .collect::<String>()
        + "END\nEND\n";
    let mut checks = vec![
        Check::Unknown { id: outside },
        Check::Served { id: no_road.clone(), mode: "car" },
        Check::Reason { id: no_road.clone(), mode: "bike", reason: "NoRoad" },
        Check::Reason { id: no_road, mode: "walk", reason: "NoRoad" },
        Check::Restrictions { at_least: 4 },
    ];
    checks.extend(on_via.into_iter().map(|id| Check::OnVia { id }));
    Scenario {
        name: "viaway",
        osm: b.osm,
        release: b.release,
        extent: Some(extent),
        variants: vec![Variant { name: "default", build_toml: build_toml(1 << 20, 1 << 30, 0, None) }],
        checks,
    }
}

/// A via-way restriction whose prefix copy has no exit (SPEC §5), as found in Lyon: A enters
/// V1, V1 enters V2, and V2's only exit X is forbidden after A, V1, V2. V1 is entered only
/// from A, so its original is reached by no turn; V2 is also entered from B. An address on
/// V2 is reached fastest through the restricted prefix, and must get that exact time.
fn deadend() -> Scenario {
    let mut b = Builder::new(Frame::new(46.0, 4.0), 6);
    let res = [("highway", "residential")];
    let one = [("highway", "residential"), ("oneway", "yes")];
    let a0 = b.node(0.0, 0.0);
    let n1 = b.node(100.0, 0.0);
    let n2 = b.node(200.0, 0.0);
    let n3 = b.node(300.0, 0.0);
    let n4 = b.node(300.0, 100.0);
    let m2 = b.node(200.0, 100.0);
    let m1 = b.node(100.0, 100.0);
    let m0 = b.node(0.0, 100.0);
    let a = b.way(&[a0, n1], &one);
    let v1 = b.way(&[n1, n2], &one);
    let v2 = b.way(&[n2, n3], &one);
    let x = b.way(&[n3, n4], &one);
    b.way(&[n2, m2], &res);
    b.way(&[n4, m2, m1, m0, a0], &res);
    b.restriction("no_straight_on", a, Via::Ways(vec![v1, v2]), x);
    taxonomy(&mut b);
    let src = b.address(50.0, 5.0);
    let on_v2 = b.address(280.0, 5.0);
    for k in 0..4 {
        b.address(20.0 + 80.0 * k as f64, 105.0);
    }
    Scenario {
        name: "deadend",
        osm: b.osm,
        release: b.release,
        extent: None,
        variants: vec![Variant { name: "default", build_toml: build_toml(1 << 20, 1 << 30, 0, None) }],
        checks: vec![
            Check::Restrictions { at_least: 1 },
            Check::DeadEnds { at_least: 1 },
            Check::OnVia { id: on_v2.clone() },
            Check::Reachable { mode: "car", a: src, b: on_v2 },
        ],
    }
}

/// A street grid with a disconnected 2-node fragment beside it, and two islands (SPEC §6):
/// an ID 5 m from the fragment and 20 m from the street snaps to the street, a larger
/// component within the radius; an ID on a large island 1.5 km away stays on it; an ID on a
/// small island 6 km away, beyond every radius, stays served there too.
fn fragment() -> Scenario {
    let mut b = Builder::new(Frame::new(47.0, 2.0), 7);
    let res = [("highway", "residential")];
    // A 4 × 3 grid, 100 m apart: 12 nodes, more than any profile's small-component size.
    let streets = |b: &mut Builder, x0: f64| {
        let g = grid(b, x0, 0.0, 4, 3, 100.0);
        for row in &g {
            b.way(row, &res);
        }
        for j in 0..3 {
            b.way(&g.iter().map(|c| c[j]).collect::<Vec<i64>>(), &res);
        }
    };
    streets(&mut b, 0.0);
    // The fragment: 40 m of road 25 m north of the street, midway between two cross streets,
    // sharing no node with them.
    let (f0, f1) = (b.node(130.0, 25.0), b.node(170.0, 25.0));
    b.way(&[f0, f1], &res);
    streets(&mut b, 1500.0);
    // The small island: a 100 m loop of four nodes, 4 km beyond the large one.
    let i = [b.node(6000.0, 0.0), b.node(6100.0, 0.0), b.node(6100.0, 100.0), b.node(6000.0, 100.0)];
    b.way(&[i[0], i[1], i[2], i[3], i[0]], &res);
    taxonomy(&mut b);
    let street = b.address(250.0, 5.0);
    let near_fragment = b.address(150.0, 20.0);
    let on_island = b.address(1550.0, 5.0);
    let on_islet = b.address(6050.0, 5.0);
    let mut checks = vec![
        Check::Reachable { mode: "car", a: street.clone(), b: near_fragment.clone() },
        Check::Reachable { mode: "walk", a: street.clone(), b: near_fragment.clone() },
        Check::Reachable { mode: "bike", a: near_fragment, b: street.clone() },
    ];
    for mode in ["car", "bike", "walk"] {
        for id in [&on_island, &on_islet] {
            checks.push(Check::Served { id: id.clone(), mode });
            checks.push(Check::DifferentComponent { mode, a: street.clone(), b: id.clone() });
        }
        checks.push(Check::DifferentComponent { mode, a: on_island.clone(), b: on_islet.clone() });
    }
    Scenario {
        name: "fragment",
        osm: b.osm,
        release: b.release,
        extent: None,
        variants: vec![Variant { name: "default", build_toml: build_toml(1 << 20, 1 << 30, 0, None) }],
        checks,
    }
}

/// A 24 × 24 grid that a small memory budget splits into cells (SPEC §5), built both ways.
fn cells() -> Scenario {
    let mut b = Builder::new(Frame::new(45.0, 5.0), 5);
    const N: usize = 24;
    let g = grid(&mut b, 0.0, 0.0, N, N, 80.0);
    for j in 0..N {
        let refs: Vec<i64> = (0..N).map(|i| g[i][j]).collect();
        let t: &[(&str, &str)] = if j % 5 == 2 { &[("highway", "residential"), ("oneway", "yes")] } else { &[("highway", "residential")] };
        b.way(&refs, t);
    }
    for (i, gi) in g.iter().enumerate() {
        let t: &[(&str, &str)] = if i % 7 == 3 { &[("highway", "tertiary"), ("oneway", "-1")] } else { &[("highway", "residential")] };
        b.way(gi, t);
    }
    taxonomy(&mut b);
    for i in 0..N {
        for j in 0..N {
            if (i + 2 * j) % 3 == 0 {
                b.address(i as f64 * 80.0 + 25.0, j as f64 * 80.0 + 9.0);
            }
            if (i * j) % 11 == 5 {
                b.place(i as f64 * 80.0 + 9.0, j as f64 * 80.0 + 50.0, Some("restaurant"), Some("restaurant"), "Resto");
            }
        }
    }
    Scenario {
        name: "cells",
        osm: b.osm,
        release: b.release,
        extent: None,
        variants: vec![
            Variant { name: "default", build_toml: build_toml(1 << 20, 1 << 30, 0, None) },
            Variant { name: "cells", build_toml: build_toml(1 << 20, 256 << 10, 0, None) },
        ],
        checks: vec![Check::Cells { family: "car" }, Check::Cells { family: "active" }],
    }
}

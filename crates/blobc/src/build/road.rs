//! The road part's units (FORMAT §5): a family's tables (§5.1, and the ACTIVE cross table)
//! and its packs (§5.2–§5.4). Every layout id written is the family's (FORMAT §4).

use blobmap_format::{NONE, Tensor, Unit};

use crate::build::family::Family;
use crate::build::hier::Built;
use crate::build::layout::Layout;

/// The family tables: the component of every layout id, and the block table.
pub fn tables(f: &dyn Family, b: &Built, l: &Layout) -> Unit {
    let mut u = Unit::default();
    let xs = &b.components;
    let mut component = Vec::with_capacity(f.nodes());
    for (x, eb) in xs.iter().enumerate() {
        component.extend(std::iter::repeat_n(x as u32, eb.nodes.len()));
    }
    u.put_in(&format!("road/{}/nodes", f.name()), "component", Tensor::vec(&component));
    let (mut kind, mut id_first, mut nodes, mut pack, mut row_first, mut levels, mut level_base) =
        (vec![], vec![], vec![], vec![], vec![], vec![], vec![]);
    for (x, eb) in xs.iter().enumerate() {
        for (k, bl) in eb.h.blocks.iter().enumerate() {
            let pl = l.place[x][k];
            kind.push(bl.kind as u8);
            id_first.push(b.first[x] + bl.id_first);
            nodes.push(bl.nodes);
            pack.push(pl.pack);
            row_first.push(pl.row_first);
            levels.push(bl.levels());
            level_base.push(pl.level_base);
        }
    }
    let p = format!("road/{}/blocks", f.name());
    u.put_in(&p, "kind", Tensor::vec(&kind));
    u.put_in(&p, "id_first", Tensor::vec(&id_first));
    u.put_in(&p, "nodes", Tensor::vec(&nodes));
    u.put_in(&p, "pack", Tensor::vec(&pack));
    u.put_in(&p, "row_first", Tensor::vec(&row_first));
    u.put_in(&p, "levels", Tensor::vec(&levels));
    u.put_in(&p, "level_base", Tensor::vec(&level_base));
    f.table_tensors(&mut u);
    u
}

pub fn pack(f: &dyn Family, b: &Built, l: &Layout, k: usize) -> Unit {
    let mut u = Unit::default();
    let p = format!("road/{}/p{k}", f.name());
    let pk = &l.packs[k];
    let metrics = f.metrics();
    let mut level_first = Vec::new();
    let (mut up_first, mut up_head) = (vec![0u32], Vec::new());
    let mut up: Vec<Vec<u16>> = vec![Vec::new(); metrics.len()];
    let mut down: Vec<Vec<u16>> = vec![Vec::new(); metrics.len()];
    for &(x, blk) in &pk.blocks {
        let e = &b.components[x as usize];
        let bl = &e.h.blocks[blk as usize];
        let base = b.first[x as usize];
        level_first.extend(bl.level_first.iter().map(|&l| base + l));
        for v in bl.id_first..bl.id_first + bl.nodes {
            let r = e.h.row(v);
            up_head.extend(e.h.up_head[r.clone()].iter().map(|&h| base + h));
            for m in 0..metrics.len() {
                up[m].extend_from_slice(&e.up[m][r.clone()]);
                down[m].extend_from_slice(&e.down[m][r.clone()]);
            }
            up_first.push(up_head.len() as u32);
        }
    }
    u.put_in(&p, "level_first", Tensor::vec(&level_first));
    u.put_in(&p, "up_first", Tensor::vec(&up_first));
    u.put_in(&p, "up_head", Tensor::vec(&up_head));
    for (m, name) in metrics.iter().enumerate() {
        u.put_in(&p, &format!("{name}.up"), Tensor::vec(&up[m]));
        u.put_in(&p, &format!("{name}.down"), Tensor::vec(&down[m]));
    }
    let e = f.components();
    let w = f.arc_word();
    let heads: Vec<u32> = pk.arcs.iter().map(|&a| b.layout_of[f.arc(a as usize).1 as usize]).collect();
    let links: Vec<u32> = pk
        .arcs
        .iter()
        .map(|&a| {
            let (t, h) = f.arc(a as usize);
            if t == h {
                return NONE;
            }
            l.link(b, e.of[t as usize], b.layout_of[t as usize], b.layout_of[h as usize])
        })
        .collect();
    u.put_in(&p, &format!("{w}_first"), Tensor::vec(&pk.arc_first));
    u.put_in(&p, &format!("{w}_head"), Tensor::vec(&heads));
    u.put_in(&p, &format!("{w}_link"), Tensor::vec(&links));
    f.row_tensors(&pk.rows, &b.layout_of, &mut u, &p);
    f.arc_tensors(&pk.arcs, &mut u, &p);
    u
}

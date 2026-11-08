//! The hashes of FORMAT §9.

use sha2::{Digest, Sha256};

use crate::dtype::Dtype;

/// A SHA-256, raw.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct Hash(pub [u8; 32]);

impl Hash {
    pub fn hex(&self) -> String {
        hex::encode(self.0)
    }

    pub fn from_hex(s: &str) -> Option<Hash> {
        let v = hex::decode(s).ok()?;
        Some(Hash(v.try_into().ok()?))
    }

    pub fn of(bytes: &[u8]) -> Hash {
        Hash(Sha256::digest(bytes).into())
    }
}

impl std::fmt::Debug for Hash {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.hex())
    }
}

/// A SHA-256 over a sequence of parts, each a byte string (FORMAT §1: ‖ is concatenation).
pub struct Hasher(Sha256);

impl Hasher {
    /// Starts with a domain tag and its 0x00 separator.
    pub fn tagged(tag: &str) -> Hasher {
        let mut h = Sha256::new();
        h.update(tag.as_bytes());
        h.update([0u8]);
        Hasher(h)
    }

    pub fn bytes(&mut self, b: &[u8]) -> &mut Self {
        self.0.update(b);
        self
    }

    /// A string followed by 0x00.
    pub fn str0(&mut self, s: &str) -> &mut Self {
        self.0.update(s.as_bytes());
        self.0.update([0u8]);
        self
    }

    pub fn hash(&mut self, h: &Hash) -> &mut Self {
        self.0.update(h.0);
        self
    }

    pub fn finish(self) -> Hash {
        Hash(self.0.finalize().into())
    }
}

/// Tensor hash = sha256("blobmap-tensor-2" ‖ 0x00 ‖ name ‖ 0x00 ‖ dtype ‖ 0x00 ‖ rank U8 ‖
/// dims U64 ‖ data). The data may come in pieces (chunks).
pub fn tensor_hash<'a>(name: &str, dtype: Dtype, shape: &[u64], data: impl IntoIterator<Item = &'a [u8]>) -> Hash {
    let mut h = Hasher::tagged("blobmap-tensor-2");
    h.str0(name).str0(dtype.name()).bytes(&[shape.len() as u8]);
    for d in shape {
        h.bytes(&d.to_le_bytes());
    }
    for piece in data {
        h.bytes(piece);
    }
    h.finish()
}

/// A hash over a tag and a list of hashes (entry, road part, vocabulary part, model ID).
pub fn of_hashes<'a>(tag: &str, hashes: impl IntoIterator<Item = &'a Hash>) -> Hash {
    let mut h = Hasher::tagged(tag);
    for x in hashes {
        h.hash(x);
    }
    h.finish()
}

pub const ENTRY_TAG: &str = "blobmap-entry-2";
pub const ROAD_TAG: &str = "blobmap-road-2";
pub const VOCAB_TAG: &str = "blobmap-vocab-2";
pub const MODEL_TAG: &str = "blobmap-model-2";

/// `model_id` (FORMAT §9).
pub fn model_id(road: &Hash, vocab: &Hash) -> Hash {
    of_hashes(MODEL_TAG, [road, vocab])
}

/// What `inputs_hash` is computed from (FORMAT §9).
pub struct Inputs<'a> {
    pub pbf_sha256: Hash,
    pub overture_release: &'a str,
    pub car_profile: Hash,
    pub active_profile: Hash,
    pub place_rules: Hash,
    pub extent: Option<Hash>,
    pub spec_version: &'a str,
}

pub fn inputs_hash(i: &Inputs) -> Hash {
    let mut h = Hasher::tagged("blobmap-inputs-2");
    h.hash(&i.pbf_sha256).str0(i.overture_release);
    h.hash(&i.car_profile).hash(&i.active_profile).hash(&i.place_rules);
    h.hash(&i.extent.unwrap_or_default());
    h.str0(i.spec_version);
    h.finish()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tensor_hash_ignores_chunking() {
        let data: Vec<u8> = (0..100).collect();
        let whole = tensor_hash("a/b", Dtype::U8, &[100], [&data[..]]);
        let pieces = tensor_hash("a/b", Dtype::U8, &[100], [&data[..37], &data[37..]]);
        assert_eq!(whole, pieces);
        assert_ne!(whole, tensor_hash("a/c", Dtype::U8, &[100], [&data[..]]));
        assert_ne!(whole, tensor_hash("a/b", Dtype::U8, &[50, 2], [&data[..]]));
    }
}

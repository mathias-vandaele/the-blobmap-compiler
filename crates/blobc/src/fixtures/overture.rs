//! A writer for the subset of an Overture release that `blobc` reads: `raw/address.parquet`,
//! `raw/place.parquet` (same column names and types as the real files) and the taxonomy CSVs.
//! Points are WKB, x = lon, y = lat, as in Overture's GeoParquet.

use std::fs::File;
use std::path::Path;
use std::sync::Arc;

use anyhow::{Context, Result};
use arrow_array::builder::{BinaryBuilder, Float64Builder, ListBuilder, StringBuilder, StructBuilder};
use arrow_array::{ArrayRef, RecordBatch, StructArray};
use arrow_schema::{DataType, Field, Fields, Schema};
use parquet::arrow::ArrowWriter;

#[derive(Clone, Debug)]
pub struct Address {
    pub id: String,
    pub lat: f64,
    pub lon: f64,
    pub country: String,
    pub dataset: String,
    pub license: String,
}

#[derive(Clone, Debug)]
pub struct Place {
    pub id: String,
    pub lat: f64,
    pub lon: f64,
    pub name: Option<String>,
    pub confidence: Option<f64>,
    pub operating_status: Option<String>,
    pub basic_category: Option<String>,
    /// `taxonomy.primary`.
    pub primary: Option<String>,
    pub alternates: Vec<String>,
    pub country: Option<String>,
    pub dataset: String,
    pub license: String,
}

/// One row of `taxonomy.csv`: the key, its hierarchy from the root, whether it is basic,
/// and its basic category.
#[derive(Clone, Debug)]
pub struct Category {
    pub hierarchy: Vec<String>,
    pub is_basic: bool,
    pub basic: Option<String>,
}

/// One redirect of `mapping_sheet.csv`: an old key, and the key it goes to (None = removed
/// without a target).
#[derive(Clone, Debug)]
pub struct Redirect {
    pub old: String,
    pub new: Option<String>,
}

#[derive(Clone, Debug, Default)]
pub struct Release {
    pub addresses: Vec<Address>,
    pub places: Vec<Place>,
    pub categories: Vec<Category>,
    pub redirects: Vec<Redirect>,
}

pub fn wkb_point(lat: f64, lon: f64) -> Vec<u8> {
    let mut b = vec![1u8];
    b.extend_from_slice(&1u32.to_le_bytes());
    b.extend_from_slice(&lon.to_le_bytes());
    b.extend_from_slice(&lat.to_le_bytes());
    b
}

fn utf8(name: &str) -> Field {
    Field::new(name, DataType::Utf8, true)
}

fn sources_field() -> Field {
    let item = Field::new("item", DataType::Struct(Fields::from(vec![utf8("dataset"), utf8("license")])), true);
    Field::new("sources", DataType::List(Arc::new(item)), true)
}

fn sources_builder() -> ListBuilder<StructBuilder> {
    let fields = Fields::from(vec![utf8("dataset"), utf8("license")]);
    let item = Field::new("item", DataType::Struct(fields.clone()), true);
    ListBuilder::new(StructBuilder::from_fields(fields, 0)).with_field(Arc::new(item))
}

fn push_source(b: &mut ListBuilder<StructBuilder>, dataset: &str, license: &str) {
    let s = b.values();
    s.field_builder::<StringBuilder>(0).unwrap().append_value(dataset);
    s.field_builder::<StringBuilder>(1).unwrap().append_value(license);
    s.append(true);
    b.append(true);
}

fn write_parquet(path: &Path, schema: Arc<Schema>, columns: Vec<ArrayRef>) -> Result<()> {
    let batch = RecordBatch::try_new(schema.clone(), columns)?;
    let f = File::create(path).with_context(|| format!("creating {}", path.display()))?;
    let mut w = ArrowWriter::try_new(f, schema, None)?;
    w.write(&batch)?;
    w.close()?;
    Ok(())
}

fn csv_field(s: &str) -> String {
    if s.contains([',', '"', '\n']) { format!("\"{}\"", s.replace('"', "\"\"")) } else { s.to_string() }
}

impl Release {
    /// Writes `raw/` and `taxonomy/` under `dir`.
    pub fn write(&self, dir: &Path) -> Result<()> {
        std::fs::create_dir_all(dir.join("raw"))?;
        std::fs::create_dir_all(dir.join("taxonomy"))?;
        self.write_addresses(&dir.join("raw/address.parquet"))?;
        self.write_places(&dir.join("raw/place.parquet"))?;
        self.write_taxonomy(&dir.join("taxonomy"))
    }

    fn write_addresses(&self, path: &Path) -> Result<()> {
        let schema =
            Arc::new(Schema::new(vec![utf8("id"), utf8("country"), Field::new("geometry", DataType::Binary, true), sources_field()]));
        let (mut id, mut country, mut geom, mut src) =
            (StringBuilder::new(), StringBuilder::new(), BinaryBuilder::new(), sources_builder());
        for a in &self.addresses {
            id.append_value(&a.id);
            country.append_value(&a.country);
            geom.append_value(wkb_point(a.lat, a.lon));
            push_source(&mut src, &a.dataset, &a.license);
        }
        let cols: Vec<ArrayRef> = vec![Arc::new(id.finish()), Arc::new(country.finish()), Arc::new(geom.finish()), Arc::new(src.finish())];
        write_parquet(path, schema, cols)
    }

    fn write_places(&self, path: &Path) -> Result<()> {
        let list_utf8 = || DataType::List(Arc::new(Field::new("item", DataType::Utf8, true)));
        let tax_fields =
            Fields::from(vec![utf8("primary"), Field::new("hierarchy", list_utf8(), true), Field::new("alternates", list_utf8(), true)]);
        let addr_fields = Fields::from(vec![utf8("country")]);
        let addr_item = Field::new("item", DataType::Struct(addr_fields.clone()), true);
        let schema = Arc::new(Schema::new(vec![
            utf8("id"),
            Field::new("geometry", DataType::Binary, true),
            Field::new("confidence", DataType::Float64, true),
            Field::new("names", DataType::Struct(Fields::from(vec![utf8("primary")])), true),
            sources_field(),
            utf8("operating_status"),
            utf8("basic_category"),
            Field::new("taxonomy", DataType::Struct(tax_fields.clone()), true),
            Field::new("addresses", DataType::List(Arc::new(addr_item.clone())), true),
        ]));
        let mut id = StringBuilder::new();
        let mut geom = BinaryBuilder::new();
        let mut conf = Float64Builder::new();
        let mut names = StructBuilder::from_fields(Fields::from(vec![utf8("primary")]), 0);
        let mut src = sources_builder();
        let mut status = StringBuilder::new();
        let mut basic = StringBuilder::new();
        let (mut primary, mut hierarchy, mut alternates) =
            (StringBuilder::new(), ListBuilder::new(StringBuilder::new()), ListBuilder::new(StringBuilder::new()));
        let mut addrs = ListBuilder::new(StructBuilder::from_fields(addr_fields, 0)).with_field(Arc::new(addr_item));
        for p in &self.places {
            id.append_value(&p.id);
            geom.append_value(wkb_point(p.lat, p.lon));
            conf.append_option(p.confidence);
            names.field_builder::<StringBuilder>(0).unwrap().append_option(p.name.as_deref());
            names.append(true);
            push_source(&mut src, &p.dataset, &p.license);
            status.append_option(p.operating_status.as_deref());
            basic.append_option(p.basic_category.as_deref());
            primary.append_option(p.primary.as_deref());
            if let Some(pr) = &p.primary {
                hierarchy.values().append_value(pr);
            }
            hierarchy.append(true);
            for x in &p.alternates {
                alternates.values().append_value(x);
            }
            alternates.append(true);
            match &p.country {
                Some(c) => {
                    let s = addrs.values();
                    s.field_builder::<StringBuilder>(0).unwrap().append_value(c);
                    s.append(true);
                    addrs.append(true);
                }
                None => addrs.append(false),
            }
        }
        let cols: Vec<ArrayRef> = vec![
            Arc::new(id.finish()),
            Arc::new(geom.finish()),
            Arc::new(conf.finish()),
            Arc::new(names.finish()),
            Arc::new(src.finish()),
            Arc::new(status.finish()),
            Arc::new(basic.finish()),
            Arc::new(StructArray::new(
                tax_fields,
                vec![Arc::new(primary.finish()), Arc::new(hierarchy.finish()), Arc::new(alternates.finish())],
                None,
            )),
            Arc::new(addrs.finish()),
        ];
        write_parquet(path, schema, cols)
    }

    fn write_taxonomy(&self, dir: &Path) -> Result<()> {
        let mut t = String::from("taxonomy,primary,displayName,level,isBasic,basicCategory\n");
        let mut b = String::from("basic_category,display_name,taxonomy,level\n");
        for c in &self.categories {
            let key = c.hierarchy.last().unwrap();
            let h = c.hierarchy.join(" > ");
            let level = c.hierarchy.len() - 1;
            t.push_str(&format!(
                "{},{},{},{level},{},{}\n",
                csv_field(&h),
                key,
                csv_field(key),
                c.is_basic,
                c.basic.as_deref().unwrap_or("")
            ));
            if c.is_basic {
                b.push_str(&format!("{key},{},{},{level}\n", csv_field(key), csv_field(&h)));
            }
        }
        let mut m = String::from(
            "Group (L0),Level,Is Basic,New Primary Hierarchy,New Primary Category,New Display Name,New Basic Level Category,\
             Old Primary Hierarchy,Old Primary Category,Old Basic Level Category,PC Added,PC Hierarchy Change,PC Name Change,\
             PC Removed,PC Redirect To,BLC Change\n",
        );
        for r in &self.redirects {
            // A removed key: "PC Removed" = TRUE, "PC Redirect To" = the target or "n/a".
            let to = r.new.as_deref().unwrap_or("n/a");
            m.push_str(&format!(",,,,,,,,{},,,,,TRUE,{to},\n", r.old));
        }
        std::fs::write(dir.join("taxonomy.csv"), t)?;
        std::fs::write(dir.join("basic_categories.csv"), b)?;
        std::fs::write(dir.join("mapping_sheet.csv"), m)?;
        Ok(())
    }
}

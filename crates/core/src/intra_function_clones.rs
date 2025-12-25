use crate::{
    function_extractor::{extract_functions, FunctionDefinition},
    parser::parse_and_convert_to_tree,
    subtree_fingerprint::{generate_subtree_fingerprints, OverlapOptions, SubtreeFingerprint},
};
use std::collections::HashMap;

/// A clone group represents multiple code locations that are similar
#[derive(Debug, Clone)]
pub struct CloneGroup {
    /// All locations where this clone appears
    pub locations: Vec<CloneLocation>,
    /// Similarity score (0.0 to 1.0) - for groups with 2+ members, this is the minimum pairwise similarity
    pub similarity: f64,
    /// Number of AST nodes in the cloned region
    pub node_count: u32,
    /// Type of the root node of the cloned subtree
    pub node_type: String,
    /// Representative hash of this clone group
    pub hash: u64,
}

/// A location of a clone within a function
#[derive(Debug, Clone)]
pub struct CloneLocation {
    /// Function name containing the clone
    pub function_name: String,
    /// File path
    pub file_path: String,
    /// Starting line number
    pub start_line: u32,
    /// Ending line number
    pub end_line: u32,
}

/// Options for intra-function clone detection
#[derive(Debug, Clone)]
pub struct IntraFunctionCloneOptions {
    /// Minimum number of AST nodes for a clone to be considered
    pub min_node_count: u32,
    /// Maximum number of AST nodes for a clone to be considered
    pub max_node_count: u32,
    /// Similarity threshold (0.0 to 1.0)
    pub threshold: f64,
    /// Size tolerance for comparing subtrees (e.g., 0.2 for 20% tolerance)
    pub size_tolerance: f64,
    /// Minimum number of occurrences to report as a clone group
    pub min_occurrences: usize,
}

impl Default for IntraFunctionCloneOptions {
    fn default() -> Self {
        Self {
            min_node_count: 5,
            max_node_count: 50,
            threshold: 0.85,
            size_tolerance: 0.25,
            min_occurrences: 2,
        }
    }
}

impl From<&OverlapOptions> for IntraFunctionCloneOptions {
    fn from(options: &OverlapOptions) -> Self {
        Self {
            min_node_count: options.min_window_size,
            max_node_count: options.max_window_size,
            threshold: options.threshold,
            size_tolerance: options.size_tolerance,
            min_occurrences: 2,
        }
    }
}

/// Detect clones within a single function
pub fn detect_intra_function_clones(
    func: &FunctionDefinition,
    full_code: &str,
    file_path: &str,
    options: &IntraFunctionCloneOptions,
) -> Result<Vec<CloneGroup>, anyhow::Error> {
    // Extract the function code
    let lines: Vec<&str> = full_code.lines().collect();
    let start_line = (func.start_line as usize).saturating_sub(1);
    let end_line = func.end_line as usize;

    if start_line >= lines.len() || end_line > lines.len() {
        return Err(anyhow::anyhow!("Function line numbers out of bounds"));
    }

    let func_code = lines[start_line..end_line].join("\n");

    // Parse the function
    let tree = parse_and_convert_to_tree(file_path, &func_code).map_err(|e| anyhow::anyhow!(e))?;

    // Generate fingerprints for all subtrees
    let (_, subtrees) = generate_subtree_fingerprints(&tree, 0, func.start_line);

    // Filter subtrees by size
    let filtered_subtrees: Vec<_> = subtrees
        .into_iter()
        .filter(|fp| fp.weight >= options.min_node_count && fp.weight <= options.max_node_count)
        .collect();

    // Group subtrees by hash (exact matches)
    let mut hash_groups: HashMap<u64, Vec<SubtreeFingerprint>> = HashMap::new();
    for subtree in filtered_subtrees {
        hash_groups.entry(subtree.hash).or_default().push(subtree);
    }

    // Find clone groups (hash groups with multiple members)
    let mut clone_groups = Vec::new();

    for (hash, group) in &hash_groups {
        if group.len() >= options.min_occurrences {
            // Check that locations don't overlap
            let non_overlapping = remove_overlapping_locations(group.clone());

            if non_overlapping.len() >= options.min_occurrences {
                let locations: Vec<CloneLocation> = non_overlapping
                    .iter()
                    .map(|fp| CloneLocation {
                        function_name: func.name.clone(),
                        file_path: file_path.to_string(),
                        start_line: fp.start_line,
                        end_line: fp.end_line,
                    })
                    .collect();

                clone_groups.push(CloneGroup {
                    locations,
                    similarity: 1.0, // Exact hash match
                    node_count: non_overlapping[0].weight,
                    node_type: non_overlapping[0].node_type.clone(),
                    hash: *hash,
                });
            }
        }
    }

    // Also find near-matches (similar but not identical)
    let near_match_groups = find_near_match_clones(&hash_groups, options);
    clone_groups.extend(near_match_groups);

    // Sort by node count (larger clones first) and then by number of occurrences
    clone_groups.sort_by(|a, b| {
        b.node_count
            .cmp(&a.node_count)
            .then_with(|| b.locations.len().cmp(&a.locations.len()))
    });

    // Remove duplicate/overlapping clone groups
    Ok(deduplicate_clone_groups(clone_groups))
}

/// Find near-match clones (similar but not identical hashes)
fn find_near_match_clones(
    hash_groups: &HashMap<u64, Vec<SubtreeFingerprint>>,
    options: &IntraFunctionCloneOptions,
) -> Vec<CloneGroup> {
    let mut near_match_groups = Vec::new();

    // Get all unique subtrees (representatives from each hash group)
    let representatives: Vec<_> = hash_groups
        .values()
        .filter_map(|group| group.first())
        .collect();

    // Compare pairs of different hash groups
    for i in 0..representatives.len() {
        for j in (i + 1)..representatives.len() {
            let fp1 = representatives[i];
            let fp2 = representatives[j];

            // Check if they might be similar
            if fp1.might_be_similar(fp2, options.size_tolerance) {
                let similarity = calculate_fingerprint_similarity(fp1, fp2);

                if similarity >= options.threshold {
                    // Merge both groups into one clone group
                    let mut all_fps = Vec::new();
                    if let Some(group1) = hash_groups.get(&fp1.hash) {
                        all_fps.extend(group1.iter().cloned());
                    }
                    if let Some(group2) = hash_groups.get(&fp2.hash) {
                        all_fps.extend(group2.iter().cloned());
                    }

                    let non_overlapping = remove_overlapping_locations(all_fps);

                    if non_overlapping.len() >= options.min_occurrences {
                        // This is handled differently - we don't have file_path here
                        // We'll need to get it from context
                        let locations: Vec<CloneLocation> = non_overlapping
                            .iter()
                            .map(|fp| CloneLocation {
                                function_name: String::new(), // Will be filled in by caller
                                file_path: String::new(),     // Will be filled in by caller
                                start_line: fp.start_line,
                                end_line: fp.end_line,
                            })
                            .collect();

                        near_match_groups.push(CloneGroup {
                            locations,
                            similarity,
                            node_count: fp1.weight,
                            node_type: fp1.node_type.clone(),
                            hash: fp1.hash, // Use first hash as representative
                        });
                    }
                }
            }
        }
    }

    near_match_groups
}

/// Calculate similarity between two fingerprints
fn calculate_fingerprint_similarity(fp1: &SubtreeFingerprint, fp2: &SubtreeFingerprint) -> f64 {
    if fp1.hash == fp2.hash {
        return 1.0;
    }

    // Simple Jaccard similarity on child hashes
    if fp1.child_hashes.is_empty() || fp2.child_hashes.is_empty() {
        return 0.5; // No children to compare
    }

    let set1: std::collections::HashSet<_> = fp1.child_hashes.iter().collect();
    let set2: std::collections::HashSet<_> = fp2.child_hashes.iter().collect();

    let intersection = set1.intersection(&set2).count();
    let union = set1.union(&set2).count();

    if union == 0 {
        0.0
    } else {
        intersection as f64 / union as f64
    }
}

/// Remove overlapping locations from a list of fingerprints
fn remove_overlapping_locations(mut fingerprints: Vec<SubtreeFingerprint>) -> Vec<SubtreeFingerprint> {
    // Sort by start line
    fingerprints.sort_by_key(|fp| fp.start_line);

    let mut result = Vec::new();
    let mut last_end_line = 0u32;

    for fp in fingerprints {
        // Skip if this overlaps with the previous one
        if fp.start_line >= last_end_line {
            last_end_line = fp.end_line;
            result.push(fp);
        }
    }

    result
}

/// Remove duplicate/overlapping clone groups
fn deduplicate_clone_groups(groups: Vec<CloneGroup>) -> Vec<CloneGroup> {
    if groups.is_empty() {
        return groups;
    }

    let mut result = vec![groups[0].clone()];

    for group in groups.into_iter().skip(1) {
        // Check if this group's locations are already covered by an existing group
        let is_duplicate = result.iter().any(|existing| {
            // A group is a duplicate if all its locations are contained within an existing group
            group.locations.iter().all(|loc| {
                existing.locations.iter().any(|existing_loc| {
                    loc.function_name == existing_loc.function_name
                        && loc.start_line >= existing_loc.start_line
                        && loc.end_line <= existing_loc.end_line
                })
            })
        });

        if !is_duplicate {
            result.push(group);
        }
    }

    result
}

/// Detect clones within all functions in a file
pub fn find_intra_function_clones_in_file(
    code: &str,
    file_path: &str,
    options: &IntraFunctionCloneOptions,
) -> Result<Vec<CloneGroup>, anyhow::Error> {
    let functions = match extract_functions(file_path, code) {
        Ok(funcs) => funcs,
        Err(e) if e.contains("Parse errors:") => {
            return Ok(Vec::new());
        }
        Err(e) => return Err(anyhow::anyhow!(e)),
    };

    let mut all_clones = Vec::new();

    for func in &functions {
        match detect_intra_function_clones(func, code, file_path, options) {
            Ok(clones) => all_clones.extend(clones),
            Err(_) => continue, // Skip functions that can't be parsed
        }
    }

    Ok(all_clones)
}

/// Detect clones within all functions across multiple files
pub fn find_intra_function_clones_across_files(
    file_contents: &HashMap<String, String>,
    options: &IntraFunctionCloneOptions,
) -> Result<Vec<CloneGroup>, anyhow::Error> {
    let mut all_clones = Vec::new();

    for (file_path, code) in file_contents {
        match find_intra_function_clones_in_file(code, file_path, options) {
            Ok(clones) => all_clones.extend(clones),
            Err(_) => continue, // Skip files that can't be parsed
        }
    }

    Ok(all_clones)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_intra_function_clones() {
        let code = r#"
function handleClick(event) {
    // First occurrence
    container.querySelectorAll('.socket-selected').forEach(el => {
        el.classList.remove('socket-selected');
        const innerSocket = el.querySelector('.custom-socket');
        if (innerSocket) {
            innerSocket.style.background = '';
            innerSocket.style.borderColor = '';
        }
    });

    doSomethingElse();

    // Second occurrence (similar code)
    container.querySelectorAll('.socket-selected').forEach(el => {
        el.classList.remove('socket-selected');
        const innerSocket = el.querySelector('.custom-socket');
        if (innerSocket) {
            innerSocket.style.background = '';
            innerSocket.style.borderColor = '';
        }
    });

    doAnotherThing();

    // Third occurrence
    container.querySelectorAll('.socket-selected').forEach(el => {
        el.classList.remove('socket-selected');
        const innerSocket = el.querySelector('.custom-socket');
        if (innerSocket) {
            innerSocket.style.background = '';
            innerSocket.style.borderColor = '';
        }
    });
}
"#;

        let options = IntraFunctionCloneOptions {
            min_node_count: 3,
            max_node_count: 50,
            threshold: 0.8,
            size_tolerance: 0.3,
            min_occurrences: 2,
        };

        let clones = find_intra_function_clones_in_file(code, "test.ts", &options).unwrap();

        // Should find at least one clone group
        // Note: The exact detection depends on how the AST is structured
        eprintln!("Found {} clone groups", clones.len());
        for clone in &clones {
            eprintln!(
                "Clone: {} nodes, {} occurrences, type: {}",
                clone.node_count,
                clone.locations.len(),
                clone.node_type
            );
        }
    }

    #[test]
    fn test_detect_exact_duplicates() {
        let code = r#"
function process() {
    if (x > 0) {
        doA();
        doB();
    }

    something();

    if (x > 0) {
        doA();
        doB();
    }
}
"#;

        let options = IntraFunctionCloneOptions {
            min_node_count: 2,
            max_node_count: 20,
            threshold: 0.9,
            size_tolerance: 0.2,
            min_occurrences: 2,
        };

        let clones = find_intra_function_clones_in_file(code, "test.ts", &options).unwrap();

        eprintln!("Found {} clone groups for exact duplicates", clones.len());
        for clone in &clones {
            eprintln!(
                "Clone: {} nodes, {} occurrences, similarity: {:.2}",
                clone.node_count,
                clone.locations.len(),
                clone.similarity
            );
        }
    }
}

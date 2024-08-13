#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <map>
#include <unordered_map>
#include <iterator>
#include <tuple>
#include <numeric>
#include <set>
#include <unordered_set>
#include <queue>
#include <iomanip>
#include <stdexcept>
#include <cassert>
#include <chrono>
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <omp.h>

const float NC_THRESH = 0.05f;
const float CONSIDERATION_THRESHOLD = 30.0f;

inline std::size_t accDictUpdate(std::unordered_map<std::string, std::size_t>& d, const std::string& acc) {
    auto it = d.find(acc);
    if (it != d.end()) {
        return it->second;
    } else {
        std::size_t n = d.size();
        d[acc] = n;
        return n;
    }
}

std::tuple<std::unordered_map<std::size_t, std::string>, std::map<std::tuple<std::size_t, std::size_t>, float>, std::size_t, std::size_t, std::vector<std::string>>
readBlastTab(std::vector<std::ifstream>& file_handles, const std::function<float(float)>& transform = nullptr) {
    std::unordered_map<std::string, std::size_t> q_accession2id, s_accession2id;
    std::unordered_map<std::size_t, std::string> id2accession;
    std::map<std::tuple<std::size_t, std::size_t>, float> similar_pairs;
    std::vector<std::string> singletons;

    std::string t;

    for (auto& fh : file_handles) {
        std::string line;
        while (std::getline(fh, line)) {
            std::istringstream iss(line);
            std::string query_id, subject_id;
            float bit_score;
            if (!(iss >> query_id >> subject_id >> t >> t >> t >> t >> t >> t >> t >> t >> t >> bit_score)) {
                throw std::runtime_error("Error parsing input: " + line);
            }

            if (subject_id == "*") {
                singletons.push_back(query_id);
            } else {
                std::size_t id1 = accDictUpdate(q_accession2id, query_id);
                std::size_t id2 = accDictUpdate(s_accession2id, subject_id);
                id2accession[id1] = query_id;
                similar_pairs[std::make_tuple(id1, id2)] = transform ? transform(bit_score) : bit_score;
            }
        }
    }
    return std::make_tuple(id2accession, similar_pairs, q_accession2id.size(), s_accession2id.size(), singletons);
}

std::tuple<std::unordered_map<std::size_t, std::string>, std::map<std::tuple<std::size_t, std::size_t>, float>, std::size_t, std::size_t>
readBlast3ColFormat(std::vector<std::ifstream>& file_handles, const std::function<float(float)>& transform = nullptr) {
    std::unordered_map<std::string, std::size_t> q_accession2id, s_accession2id;
    std::unordered_map<std::size_t, std::string> id2accession;
    std::map<std::tuple<std::size_t, std::size_t>, float> similar_pairs;

    for (auto& fh : file_handles) {
        std::string line;
        while (std::getline(fh, line)) {
            std::istringstream iss(line);
            std::string query_id, subject_id;
            float bit_score;
            if (!(iss >> query_id >> subject_id >> std::ws >> bit_score)) {
                throw std::runtime_error("Error parsing input: " + line);
            }

            std::size_t id1 = accDictUpdate(q_accession2id, query_id);
            std::size_t id2 = accDictUpdate(s_accession2id, subject_id);
            id2accession[id1] = query_id;
            similar_pairs[std::make_tuple(id1, id2)] = transform ? transform(bit_score) : bit_score;
        }
    }
    return std::make_tuple(id2accession, similar_pairs, q_accession2id.size(), s_accession2id.size());
}

float pearsonCorrelation(const Eigen::SparseMatrix<float, Eigen::RowMajor>& matrix, std::size_t i, std::size_t j,
                         std::vector<std::unordered_map<std::size_t, std::tuple<float, float, float>>>& thread_local_caches) {
    Eigen::SparseVector<float> row_i = matrix.row(i);
    Eigen::SparseVector<float> row_j = matrix.row(j);

    int thread_id = omp_get_thread_num();
    auto& cache = thread_local_caches[thread_id];

    auto get_cached_values = [&](std::size_t idx, const Eigen::VectorXf& row) {
        auto it = cache.find(idx);
        if (it != cache.end()) {
            return it->second;
        } else {
            float mean = row.mean();
            float sum = row.sum();
            float root_variance = std::sqrt(row.dot(row) - 2 * mean * sum + row.size() * mean * mean);
            cache[idx] = std::make_tuple(mean, sum, root_variance);
            return std::make_tuple(mean, sum, root_variance);
        }
    };

    auto [mean_i, sum_i, root_variance_i] = get_cached_values(i, row_i);
    auto [mean_j, sum_j, root_variance_j] = get_cached_values(j, row_j);

    float numerator = row_i.dot(row_j) - mean_i * sum_j - mean_j * sum_i + row_i.size() * mean_i * mean_j;
    float denominator = root_variance_i * root_variance_j;

    if (std::abs(denominator) < 1e-10f) return 0.0f;
    return numerator / denominator;
}

Eigen::SparseMatrix<float, Eigen::RowMajor> scoresToComparisonMatrix(const std::map<std::tuple<std::size_t, std::size_t>, float>& similar_pairs, std::size_t nrows, std::size_t ncols) {
    std::vector<Eigen::Triplet<float>> tripletList;
    tripletList.reserve(similar_pairs.size());

    for (const auto& pair : similar_pairs) {
        std::size_t row, col;
        float value;
        std::tie(row, col) = pair.first;
        value = pair.second;
        tripletList.emplace_back(row, col, value);
    }

    Eigen::SparseMatrix<float, Eigen::RowMajor> comparisonMatrix(nrows, ncols);
    comparisonMatrix.setFromTriplets(tripletList.begin(), tripletList.end());

    std::cout << "Prepared matrix with similarity data (" << nrows << " by " << ncols << ", but a sparse matrix)" << std::endl;
    std::cout << "There are " << comparisonMatrix.nonZeros() << " (=" << 100.0 * static_cast<double>(comparisonMatrix.nonZeros()) / static_cast<double>(nrows * ncols) << "%) non-zero elements in the matrix" << std::endl;

    return comparisonMatrix;
}

std::vector<std::pair<std::size_t, std::size_t>> findGoodPairs(
    const std::map<std::tuple<std::size_t, std::size_t>, float>& similarities,
    float threshold,
    bool xross) {
    
    std::unordered_map<std::size_t, std::vector<std::pair<std::size_t, std::size_t>>> ref_hits;
    
    // First pass: populate ref_hits
    for (const auto& [indices, score] : similarities) {
        auto [a, file_num] = indices;
        std::size_t b = std::get<1>(indices);
        if (score >= threshold) {
            ref_hits[b].emplace_back(a, file_num);
            // std::cout << "Adding to ref_hits: " << b << " <- (" << a << ", " << file_num << ") score: " << score << std::endl;
        }
    }

    std::set<std::pair<std::size_t, std::size_t>> good_pairs;

    // Second pass: generate good pairs
    for (const auto& [target, queries] : ref_hits) {
        for (size_t i = 0; i < queries.size(); ++i) {
            for (size_t j = i + 1; j < queries.size(); ++j) {
                auto [a, file_num_a] = queries[i];
                auto [b, file_num_b] = queries[j];
                
              //  std::cout << "Considering pair: (" << a << ", " << b << ") from files " 
              //            << file_num_a << ", " << file_num_b << std::endl;
                
                if (!xross || file_num_a != file_num_b) {
                    std::size_t min_a_b = std::min(a, b);
                    std::size_t max_a_b = std::max(a, b);
                    good_pairs.emplace(min_a_b, max_a_b);
             //       std::cout << "Adding good pair: (" << min_a_b << ", " << max_a_b << ")" << std::endl;
                }
            }
        }
    }

    return std::vector<std::pair<std::size_t, std::size_t>>(good_pairs.begin(), good_pairs.end());
}

void neighborhoodCorrelation(
        const std::unordered_map<std::size_t, std::string>& id2accession,
        const std::map<std::tuple<std::size_t, std::size_t>, float>& similar_pairs,
        std::size_t n_queries, std::size_t n_ref_seqs, float threshold, bool xross,
        std::function<void(std::size_t, std::size_t, float)> process_result) {

    auto good_pairs = findGoodPairs(similar_pairs, threshold, xross);

    Eigen::SparseMatrix<float, Eigen::RowMajor> comparison_matrix = scoresToComparisonMatrix(similar_pairs, n_queries, n_ref_seqs);

    std::cout << "similar_pairs length : " << similar_pairs.size() << std::endl;
    std::cout << "good_pairs length : " << good_pairs.size() << std::endl;
    std::cout << "starting pearson correlation" << std::endl;

    std::size_t counter = 0;
    const std::size_t print_interval = 100000;
    auto start_time = std::chrono::high_resolution_clock::now();

    int num_threads = omp_get_max_threads();
    std::vector<std::unordered_map<std::size_t, std::tuple<float, float, float>>> thread_local_caches(num_threads);

#pragma omp parallel for
    for (std::size_t k = 0; k < good_pairs.size(); ++k) {
        std::size_t a = good_pairs[k].first;
        std::size_t b = good_pairs[k].second;
        float cor = pearsonCorrelation(comparison_matrix, a, b, thread_local_caches);

#pragma omp critical
        {
            process_result(a, b, cor);
            counter++;
            if (counter % print_interval == 0) {
                auto end_time = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double> elapsed_seconds = end_time - start_time;
                std::cout << "Pairs analyzed: " << counter << " - Time elapsed: " << elapsed_seconds.count() << " seconds" << std::endl;
                start_time = std::chrono::high_resolution_clock::now();
            }
        }
    }
}

void processData(std::vector<std::ifstream>& file_handles, bool three_col, bool verbose, bool xross_files, float consider_threshold, 
                 const std::function<float(float)>& transform, const std::string& output_file) {
    std::unordered_map<std::size_t, std::string> id2accession;
    std::map<std::tuple<std::size_t, std::size_t>, float> similar_pairs;
    std::size_t n_queries = 0, n_ref_seqs = 0;
    std::vector<std::string> singletons;

    std::cout << "Reading input data..." << std::endl;

    if (three_col) {
        std::tie(id2accession, similar_pairs, n_queries, n_ref_seqs) = readBlast3ColFormat(file_handles, transform);
    } else {
        std::tie(id2accession, similar_pairs, n_queries, n_ref_seqs, singletons) = readBlastTab(file_handles, transform);
    }

    std::cout << "Finished reading input data." << std::endl;
    std::cout << "n_queries: " << n_queries << ", n_ref_seqs: " << n_ref_seqs << std::endl;
    std::cout << "similar_pairs size: " << similar_pairs.size() << std::endl;

    if (!singletons.empty() && verbose) {
        std::cerr << "Noted " << singletons.size() << " sequences without a hit in the reference data." << std::endl;
    }

    if (!similar_pairs.empty()) {
        std::cout << "Starting neighborhood correlation..." << std::endl;
        std::size_t counter = 0;

        std::ofstream out_file(output_file);
        if (!out_file.is_open()) {
            throw std::runtime_error("Unable to open output file: " + output_file);
        }

        neighborhoodCorrelation(id2accession, similar_pairs, n_queries, n_ref_seqs, consider_threshold, xross_files,
            [&](std::size_t acc1, std::size_t acc2, float nc) {
                counter++;
                if (verbose && counter % 100000 == 0) {
                    std::cerr << counter << " pairs analyzed" << std::endl;
                }
                if (nc >= NC_THRESH) {
                    out_file << id2accession[acc1] << " " << id2accession[acc2] << " " << std::fixed << std::setprecision(3) << nc << std::endl;
                }
            }
        );

        out_file.close();

        if (verbose) {
            std::cerr << counter << " pairs analyzed" << std::endl;
        }
    } else {
        std::cout << "No similar pairs found." << std::endl;
    }

    if (verbose) {
        std::cerr << "Done." << std::endl;
    }
}

float sqrtTransform(float sc) {
    return std::sqrt(sc);
}

float rootTransform(float sc, float r) {
    return static_cast<float>(std::pow(sc, 1.0f / r));
}

float log10Transform(float sc) {
    return std::log10(sc + 1.0f);
}

float lnTransform(float sc) {
    return std::log(sc + 1.0f);
}

int main(int argc, char* argv[]) {
auto start_time = std::chrono::high_resolution_clock::now();
    try {
        if (argc < 2) {
            std::cerr << "Usage: " << argv[0] << " <input_files...> [options]" << std::endl;
            std::cerr << "Options:" << std::endl;
            std::cerr << "  -o <output_file>    Specify output file (default: output.txt)" << std::endl;
            std::cerr << "  -3                  Use 3-column input format" << std::endl;
            std::cerr << "  -v                  Verbose output" << std::endl;
            std::cerr << "  -x                  Cross-file mode" << std::endl;
            std::cerr << "  -c <threshold>      Consideration threshold (default: " << CONSIDERATION_THRESHOLD << ")" << std::endl;
            std::cerr << "  -t <transform>      Score transform (sqrt, cubicroot, 2.5root, log10, ln)" << std::endl;
            return 1;
        }

        std::vector<std::string> input_files;
        std::string output_file = "output.txt";
        bool three_col = false;
        bool verbose = true;
        bool xross_files = false;
        float consider_threshold = CONSIDERATION_THRESHOLD;
        std::function<float(float)> transform = nullptr;

        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];
            if (arg[0] == '-') {
                if (arg == "-o" && i + 1 < argc) output_file = argv[++i];
                else if (arg == "-3") three_col = true;
                else if (arg == "-v") verbose = true;
                else if (arg == "-x") xross_files = true;
                else if (arg == "-c" && i + 1 < argc) consider_threshold = std::stof(argv[++i]);
                else if (arg == "-t" && i + 1 < argc) {
                    std::string transform_name = argv[++i];
                    if (transform_name == "sqrt") transform = sqrtTransform;
                    else if (transform_name == "cubicroot") transform = [](float sc) { return rootTransform(sc, 3.0f); };
                    else if (transform_name == "2.5root") transform = [](float sc) { return rootTransform(sc, 2.5f); };
                    else if (transform_name == "log10") transform = log10Transform;
                    else if (transform_name == "ln") transform = lnTransform;
                    else throw std::runtime_error("Unknown transform: " + transform_name);
                }
            } else {
                input_files.push_back(arg);
            }
        }

        if (input_files.empty()) {
            throw std::runtime_error("No input files specified");
        }

        std::vector<std::ifstream> file_handles;
        for (const auto& file : input_files) {
            file_handles.emplace_back(file);
            if (!file_handles.back().is_open()) {
                throw std::runtime_error("Failed to open file: " + file);
            }
            std::cout << "Opened file: " << file << std::endl;
        }

        processData(file_handles, three_col, verbose, xross_files, consider_threshold, transform, output_file);

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "Total execution time: " << duration.count() / 1000.0 << " seconds" << std::endl;

    return 0;
}
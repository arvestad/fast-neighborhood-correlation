#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <cmath>
#include <algorithm>
#include <tuple>
#include <string>
#include <functional>
#include <chrono>
#include <omp.h>
#include <sys/resource.h>
#include <iomanip>

#define NC_THRESH 0.05
#define CONSIDERATION_THRESHOLD 30

using namespace std;

namespace std {
    template <>
    struct hash<pair<int, int>> {
        size_t operator()(const pair<int, int>& p) const {
            return hash<int>()(p.first) ^ hash<int>()(p.second);
        }
    };
}

int acc_dict_update(unordered_map<string, int>& d, const string& acc) {
    if (d.find(acc) != d.end()) {
        return d[acc];
    } else {
        int n = d.size();
        d[acc] = n;
        return n;
    }
}

struct SparseVector {
    vector<pair<int, double>> data;

    void add(int index, double value) {
        data.emplace_back(index, value);
    }

    void sort() {
        std::sort(data.begin(), data.end());
    }
};

double sqrt_transform(double sc) {
    return sqrt(sc);
}

double root_transform(double sc, double r) {
    return pow(sc, 1.0 / r);
}

double log10_transform(double sc) {
    return log10(sc + 1);
}

double ln_transform(double sc) {
    return log(sc + 1);
}

tuple<unordered_map<int, string>, unordered_map<pair<int, int>, double>, int, int, vector<string>>
read_blast_file(vector<ifstream>& file_handles, const function<double(double)>& transform = nullptr) {
    unordered_map<string, int> q_accession2id;
    unordered_map<string, int> s_accession2id;
    unordered_map<int, string> id2accession;
    unordered_map<pair<int, int>, double> similar_pairs;
    vector<string> singletons;

    bool is_3col_format = true;

    for (size_t file_num = 0; file_num < file_handles.size(); ++file_num) {
        auto& fh = file_handles[file_num];
        string line;
        int line_no = 0;
        while (getline(fh, line)) {
            ++line_no;
            istringstream iss(line);
            string query_id, subject_id;
            double bit_score;
           
            if (line_no == 1) {
                vector<string> columns;
                string col;
                while (iss >> col) {
                    columns.push_back(col);
                }
                is_3col_format = (columns.size() == 3);
                iss.clear();
                iss.str(line);
            }

            if (is_3col_format) {
                if (!(iss >> query_id >> subject_id >> bit_score)) {
                    cerr << "Could not parse 3-column input from line " << line_no
                         << " in file " << file_num << ". Contents: \"" << line << "\"" << endl;
                    throw runtime_error("Error parsing input");
                }
            } else {
                string t;
                if (!(iss >> query_id >> subject_id >> t >> t >> t >> t >> t >> t >> t >> t >> t >> bit_score)) {
                    cerr << "Could not parse 11-column input from line " << line_no
                         << " in file " << file_num << ". Contents: \"" << line << "\"" << endl;
                    throw runtime_error("Error parsing input");
                }
                if (subject_id == "*") {
                    singletons.push_back(query_id);
                    continue;
                }
            }

            int id1 = acc_dict_update(q_accession2id, query_id);
            int id2 = acc_dict_update(s_accession2id, subject_id);
            id2accession[id1] = query_id;

            if (transform) {
                similar_pairs[{id1 * file_handles.size() + file_num, id2}] = transform(bit_score);
            } else {
                similar_pairs[{id1 * file_handles.size() + file_num, id2}] = bit_score;
            }
        }
    }

    return {id2accession, similar_pairs,
            static_cast<int>(q_accession2id.size()),
            static_cast<int>(s_accession2id.size()),
            singletons};
}

double pearson_correlation_optimized(const SparseVector& vec_a, const SparseVector& vec_b, int n_ref_seqs) {
    double sum_a = 0, sum_b = 0, sum_ab = 0, sum_a2 = 0, sum_b2 = 0;
    size_t i = 0, j = 0;

    while (i < vec_a.data.size() && j < vec_b.data.size()) {
        if (vec_a.data[i].first < vec_b.data[j].first) {
            sum_a += vec_a.data[i].second;
            sum_a2 += vec_a.data[i].second * vec_a.data[i].second;
            i++;
        } else if (vec_a.data[i].first > vec_b.data[j].first) {
            sum_b += vec_b.data[j].second;
            sum_b2 += vec_b.data[j].second * vec_b.data[j].second;
            j++;
        } else {
            sum_a += vec_a.data[i].second;
            sum_b += vec_b.data[j].second;
            sum_ab += vec_a.data[i].second * vec_b.data[j].second;
            sum_a2 += vec_a.data[i].second * vec_a.data[i].second;
            sum_b2 += vec_b.data[j].second * vec_b.data[j].second;
            i++;
            j++;
        }
    }

    while (i < vec_a.data.size()) {
        sum_a += vec_a.data[i].second;
        sum_a2 += vec_a.data[i].second * vec_a.data[i].second;
        i++;
    }

    while (j < vec_b.data.size()) {
        sum_b += vec_b.data[j].second;
        sum_b2 += vec_b.data[j].second * vec_b.data[j].second;
        j++;
    }

    double mean_a = sum_a / n_ref_seqs;
    double mean_b = sum_b / n_ref_seqs;

    double numerator = sum_ab - n_ref_seqs * mean_a * mean_b;
    double denominator = sqrt((sum_a2 - n_ref_seqs * mean_a * mean_a) * (sum_b2 - n_ref_seqs * mean_b * mean_b));

    if (denominator == 0) return 0;

    return numerator / denominator;
}

struct pair_hash {
    inline std::size_t operator()(const std::pair<int,int> & v) const {
        return v.first * 31 + v.second;
    }
};

unordered_set<pair<int, int>, pair_hash> find_good_pairs(
    const unordered_map<pair<int, int>, double>& similar_pairs,
    double threshold,
    bool xross
) {
    unordered_map<int, vector<pair<int, int>>> ref_hits;
    ref_hits.reserve(similar_pairs.size() / 10);

    unordered_map<int, int> ref_counts;
    for (const auto& [indices, score] : similar_pairs) {
        if (score >= threshold) {
            ref_counts[indices.second]++;
        }
    }

    for (const auto& [ref, count] : ref_counts) {
        ref_hits[ref].reserve(count);
    }

    for (const auto& [indices, score] : similar_pairs) {
        if (score >= threshold) {
            uint64_t value = static_cast<uint64_t>(indices.first);
            ref_hits[indices.second].emplace_back(indices.first, value >> 32);
        }
    }

    unordered_set<pair<int, int>, pair_hash> good_pairs;
    good_pairs.reserve(similar_pairs.size() / 2);

    for (const auto& [target, queries] : ref_hits) {
        for (size_t i = 0; i < queries.size(); ++i) {
            for (size_t j = i + 1; j < queries.size(); ++j) {
                if (!xross || queries[i].second != queries[j].second) {
                    int a = min(queries[i].first, queries[j].first);
                    int b = max(queries[i].first, queries[j].first);
                    good_pairs.emplace(a, b);
                }
            }
        }
    }

    return good_pairs;
}

void print_memory_usage(const string& message) {
    struct rusage r_usage;
    getrusage(RUSAGE_SELF, &r_usage);
    cerr << message << " - Memory usage: " << r_usage.ru_maxrss << " KB" << endl;
}

void neighborhood_correlation(const string& output_file, unordered_map<int, string>& id2accession,
                              unordered_map<pair<int, int>, double>& similar_pairs, int n_queries, int n_ref_seqs,
                              double threshold, bool xross, bool verbose, double nc_thresh = NC_THRESH) {
    if (verbose) cerr << "Starting neighborhood_correlation" << endl;
    auto start_time = chrono::steady_clock::now();

    auto good_pairs = find_good_pairs(similar_pairs, threshold, xross);

    if (verbose) {
        auto end_time = chrono::steady_clock::now();
        auto duration = chrono::duration_cast<chrono::milliseconds>(end_time - start_time).count();
        cerr << "find_good_pairs took " << duration << " milliseconds" << endl;
        cerr << "Good pairs calculated: " << good_pairs.size() << endl;
        cerr << "similar_pairs length : " << similar_pairs.size() << std::endl;
        print_memory_usage("After calculating good pairs");
    }

    vector<SparseVector> sparse_vectors(n_queries);
    for (const auto& [pair, score] : similar_pairs) {
        sparse_vectors[pair.first].add(pair.second, score);
    }

    #pragma omp parallel for
    for (int i = 0; i < n_queries; ++i) {
        sparse_vectors[i].sort();
    }

    if (verbose) print_memory_usage("After pre-computing sparse vectors");

    start_time = chrono::steady_clock::now();

    ofstream outfile(output_file);
    if (!outfile.is_open()) {
        cerr << "Failed to open output file: " << output_file << endl;
        return;
    }

    outfile << fixed << setprecision(3);

    int counter = 0;
    int print_interval = 100000;

    vector<pair<int, int>> good_pairs_vec(good_pairs.begin(), good_pairs.end());

    #pragma omp parallel
    {
        vector<tuple<string, string, double>> local_results;
        local_results.reserve(print_interval);

        #pragma omp for
        for (size_t i = 0; i < good_pairs_vec.size(); ++i) {
            const auto& pair = good_pairs_vec[i];
            double cor = pearson_correlation_optimized(sparse_vectors[pair.first], sparse_vectors[pair.second], n_ref_seqs);
            if (cor >= nc_thresh) {
                local_results.emplace_back(id2accession[pair.first], id2accession[pair.second], cor);
            }

            #pragma omp critical
            {
                counter++;
                if (verbose && counter % print_interval == 0) {
                    auto elapsed_time = chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - start_time).count();
                    cerr << "Processed: " << counter << "/" << good_pairs.size() << " - Time elapsed: " << elapsed_time << " ms" << endl;
                    print_memory_usage("During processing");
                }
            }
        }

        #pragma omp critical
        {
            for (const auto& [acc1, acc2, cor] : local_results) {
                outfile << acc1 << " " << acc2 << " " << cor << "\n";
            }
        }
    }

    outfile.close();
    if (verbose) {
        cerr << "Neighborhood correlation completed" << endl;
        print_memory_usage("After completion");
    }
}

int main(int argc, char* argv[]) {
    auto start_time = std::chrono::high_resolution_clock::now();

    if (argc < 2) {
        cerr << "Usage: " << argv[0] << " <input_files...> [options]" << endl;
        cerr << "Options:" << endl;
        cerr << "  -o  <output_file>    Specify output file (default: output_results.txt)" << endl;
        cerr << "  -v                   Verbose output" << endl;
        cerr << "  -x                   Only consider pairs of sequences from different files" << endl;
        cerr << "  -c  <threshold>      Consideration threshold (default: " << CONSIDERATION_THRESHOLD << ")" << endl;
	cerr << "                       Only consider pairs of sequences linked by similarities (maybe in" << endl;
	cerr << "                       several steps) with this bitscores or higher." << endl;
        cerr << "  -st <transform>      Score transform (sqrt, cubicroot, 2.5root, log10, ln)" << endl;
	cerr << "                       Transform the input bitscores with one of the given functions. " << endl;
	cerr << "                       The two logarithmic transforms are actually on the bitscore + 1," << endl;
	cerr << "                       to avoid issues around zero." << endl;
        return 1;
    }

    vector<string> input_files;
    string output_file = "output_results.txt";
    bool verbose = false;
    bool xross_files = false;
    double consider = CONSIDERATION_THRESHOLD;
    function<double(double)> transform = nullptr;

    for (int i = 1; i < argc; ++i) {
        string arg = argv[i];
        if (arg[0] == '-') {
            if (arg == "-o" && i + 1 < argc) output_file = argv[++i];
            else if (arg == "-v") verbose = true;
            else if (arg == "-x") xross_files = true;
            else if (arg == "-c" && i + 1 < argc) consider = stod(argv[++i]);
            else if (arg == "-st" && i + 1 < argc) {
                string transform_name = argv[++i];
                if (transform_name == "sqrt") transform = sqrt_transform;
                else if (transform_name == "cubicroot") transform = [](double sc) { return root_transform(sc, 3.0); };
                else if (transform_name == "2.5root") transform = [](double sc) { return root_transform(sc, 2.5); };
                else if (transform_name == "log10") transform = log10_transform;
                else if (transform_name == "ln") transform = ln_transform;
                else {
                    cerr << "Unknown transform: " << transform_name << endl;
                    return 1;
                }
            }
        } else {
            input_files.push_back(arg);
        }
    }

    if (input_files.empty()) {
        cerr << "No input files specified" << endl;
        return 1;
    }

    vector<ifstream> file_handles;
    for (const auto& file : input_files) {
        file_handles.emplace_back(file);
        if (!file_handles.back().is_open()) {
            cerr << "Could not open file: " << file << endl;
            return 1;
        }
    }

    unordered_map<int, string> id2accession;
    unordered_map<pair<int, int>, double> similarities;
    int n_queries, n_ref_seqs;
    vector<string> singletons;

    try {
        tie(id2accession, similarities, n_queries, n_ref_seqs, singletons) = read_blast_file(file_handles, transform);
    } catch (const runtime_error& e) {
        cerr << "Error reading input files: " << e.what() << endl;
        return 1;
    }

    if (!singletons.empty() && verbose) {
        cerr << "Noted " << singletons.size() << " sequences without a hit in the reference data." << endl;
    }

    if (!similarities.empty()) {
        neighborhood_correlation(output_file, id2accession, similarities, n_queries, n_ref_seqs, consider, xross_files, verbose);
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cerr << "Total execution time: " << duration.count() / 1000.0 << " seconds" << std::endl;
    return 0;
}

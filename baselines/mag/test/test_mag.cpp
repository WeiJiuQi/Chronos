#include "index_mag.h"
#include "util.h"

void save_results(std::string result_file, std::vector<std::vector<unsigned>> &kmips) {
  std::ofstream out(result_file);
  if (!out.is_open()) {
    std::cerr << "Cannot open result file: " << result_file << std::endl;
    exit(1);
  }
  for (unsigned i = 0; i < kmips.size(); i++) {
    for (unsigned j = 0; j < kmips[i].size(); j++) {
      if (j != kmips[i].size() - 1) {
        out << kmips[i][j] << " ";
      } else {
        out << kmips[i][j] << "\n";
      }
    }
  }
}


int main(int argc, char** argv) {

  std::cout << "Data Path: " << argv[1] << std::endl;

  unsigned points_num, dim=(unsigned)atoi(argv[8]);
  float* data_load = nullptr;
  data_load = MAG::load_data(argv[1], points_num, dim);
  data_load = MAG::data_align(data_load, points_num, dim);
  std::string mode(argv[7]);
  if (mode == "index") {
    std::string nn_graph_path(argv[2]);
    unsigned L = (unsigned)atoi(argv[3]);
    unsigned R = (unsigned)atoi(argv[4]);
    unsigned C = (unsigned)atof(argv[5]);
    unsigned R_IP = (unsigned)atoi(argv[9]);
    unsigned M = (unsigned)atoi(argv[10]);
    unsigned threshold = (unsigned)atoi(argv[11]);
    std::cout << "L = " << L << ", ";
    std::cout << "R = " << R << ", ";
    std::cout << "C = " << C << std::endl;
    std::cout << "KNNG = " << nn_graph_path << std::endl;
    std::cout << "R_IP = " << R_IP << std::endl;
    std::cout << "M = " << M << std::endl;
    MAG::IndexMAG index(dim, points_num);
    MAG::Parameters paras;
    paras.Set<unsigned>("L", L);
    paras.Set<unsigned>("R", R);
    paras.Set<unsigned>("C", C);
    paras.Set<unsigned>("R_IP", R_IP);
    paras.Set<unsigned>("threshold", threshold);
    paras.Set<unsigned>("n_try", 1);
    paras.Set<unsigned>("M", M);
    paras.Set<std::string>("nn_graph_path", nn_graph_path);

    std::cout << "Output ARDG Path: " << argv[6] << std::endl;

    auto s = std::chrono::high_resolution_clock::now();
    index.Build(points_num, data_load, paras);
    auto e = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = e - s;
    std::cout << "Build Time: " << diff.count() << "\n";
    index.Save(argv[6]);
  } else {
    std::cout << "Query Path: " << argv[2] << std::endl;
    unsigned query_num, query_dim = (unsigned)atoi(argv[8]);
    float* query_load = nullptr;
    query_load = MAG::load_data(argv[2], query_num, query_dim);
    query_load = MAG::data_align(query_load, query_num, query_dim);

    assert(dim == query_dim);
    MAG::IndexMAG index(dim, points_num);
    std::cout << "ARDG Path: " << argv[3] << std::endl;
    std::cout << "Result Path: " << argv[6] << std::endl;

    index.Load(argv[3]);

    unsigned L = (unsigned)atoi(argv[4]);
    unsigned K = (unsigned)atoi(argv[5]);

    std::cout << "L = " << L << ", ";
    std::cout << "K = " << K << std::endl;

    MAG::Parameters paras;
    paras.Set<unsigned>("L_search", L);

    std::vector<std::vector<unsigned> > res(query_num);
    for (unsigned i = 0; i < query_num; i++) res[i].resize(K);

    index.entry_point_candidate(data_load);

    auto s = std::chrono::high_resolution_clock::now();
    auto metric_compuations = 0;
    #pragma omp parallel for
    for (unsigned i = 0; i < query_num; i++) {
      metric_compuations += index.Search_NN_IP(query_load + i * dim, data_load, K, paras, res[i].data());
    }
    auto e = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> diff = e - s;
    std::cout << "Average query time: " << std::chrono::duration_cast<std::chrono::milliseconds>(e - s).count() / (double) query_num << "ms" << std::endl;
    std::cout << "Average metric computations: " << metric_compuations / (double) query_num << std::endl;
    save_results(argv[6], res);
    }

  return 0;
}

#include <chrono>

#include "ssg/index_random.h"
#include "mips/index_mips.h"
#include "ssg/util.h"

void save_result(char* filename, std::vector<std::vector<unsigned> >& results) {
  std::ofstream out(filename, std::ios::binary | std::ios::out);

  for (unsigned i = 0; i < results.size(); i++) {
    unsigned GK = (unsigned)results[i].size();
    out.write((char*)&GK, sizeof(unsigned));
    out.write((char*)results[i].data(), GK * sizeof(unsigned));
  }
  out.close();
}

void read_kmeans(const std::string& filename, int query_count, std::vector<int>& query_cluster, std::vector<std::vector<int>>& init_nodes_clusters, int number_of_clusters) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Cannot open file: " << filename << std::endl;
        return;
    }

    query_cluster.resize(query_count);
    file.read(reinterpret_cast<char*>(query_cluster.data()), query_count * sizeof(int));

    std::vector<int> init_nodes;
    file.seekg(0, std::ios::end);
    size_t file_size = file.tellg();
    size_t init_nodes_count = (file_size - query_count * sizeof(int)) / sizeof(int);
    file.seekg(query_count * sizeof(int), std::ios::beg);

    init_nodes.resize(init_nodes_count);
    file.read(reinterpret_cast<char*>(init_nodes.data()), init_nodes_count * sizeof(int));

    file.close();

    size_t cluster_size = number_of_clusters;
    for (size_t i = 0; i < init_nodes.size(); i += cluster_size) {
        std::vector<int> cluster(init_nodes.begin() + i, init_nodes.begin() + i + cluster_size);
        init_nodes_clusters.push_back(cluster);
    }
}

int main(int argc, char** argv) {
  if (argc < 8) {
    std::cout << "./run data_file query_file ssg_path L K result_path dim"
              << std::endl;
    exit(-1);
  }

  std::cerr << "Data Path: " << argv[1] << std::endl;

  unsigned points_num, dim = (unsigned)atoi(argv[7]);
  float* data_load = nullptr;
  data_load = efanna2e::load_data(argv[1], points_num, dim);
  data_load = efanna2e::data_align(data_load, points_num, dim);

  std::cerr << "Query Path: " << argv[2] << std::endl;

  unsigned query_num, query_dim = (unsigned)atoi(argv[7]);
  float* query_load = nullptr;
  query_load = efanna2e::load_data(argv[2], query_num, query_dim);
  query_load = efanna2e::data_align(query_load, query_num, query_dim);

  assert(dim == query_dim);

  efanna2e::IndexRandom init_index(dim, points_num);
  efanna2e::IndexMips index(dim, points_num, efanna2e::FAST_L2,
                           (efanna2e::Index*)(&init_index));

  std::cerr << "SSG Path: " << argv[3] << std::endl;
  std::cerr << "Result Path: " << argv[6] << std::endl;
  index.SaveData(data_load);
  index.Load(argv[3]);

  unsigned L = (unsigned)atoi(argv[4]);
  unsigned K = (unsigned)atoi(argv[5]);

  std::cerr << "L = " << L << ", ";
  std::cerr << "K = " << K << std::endl;

  efanna2e::Parameters paras;
  paras.Set<unsigned>("L_search", L);
  std::vector<std::vector<unsigned> > res(query_num);
  for (unsigned i = 0; i < query_num; i++) res[i].resize(K);

  /* optional entry points initialization */
  // std::string filename = "../output/mnist/sn.bin";
  // std::vector<int> query_cluster;
  // std::vector<std::vector<int>> init_nodes_clusters;
  // read_kmeans(filename, query_num, query_cluster, init_nodes_clusters, 100);

  auto num = 0.0;

  std::vector<std::pair<float,float>> cal_pair;


  auto start = std::chrono::high_resolution_clock::now();

  #pragma omp parallel for
  for (unsigned i = 0; i < (int)query_num; i++) {
    // int dis_cal = index.Search_Mips_IP_Cal(query_load + i * dim, data_load, K, paras, res[i].data(), init_nodes_clusters[query_cluster[i]]);
    int dis_cal = index.Search_Mips_IP_Cal_with_No_SN(query_load + i * dim, data_load, K, paras, res[i].data());
    num += dis_cal;
  }

  auto end = std::chrono::high_resolution_clock::now();

  std::cout << "Average Distance Computation: " << num / (int)query_num << std::endl;
  std::cout << "Average Query Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() / static_cast<double>(query_num) << "ms" << std::endl;
  save_result(argv[6], res);

  return 0;
}
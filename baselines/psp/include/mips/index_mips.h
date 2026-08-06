#pragma once

#include <boost/dynamic_bitset.hpp>
#include <cassert>
#include <sstream>
#include <stack>
#include <string>
#include <unordered_map>

#include "../ssg/index.h"
#include "../ssg/neighbor.h"
#include "../ssg/parameters.h"
#include "../ssg/util.h"
#include "../ssg/distance.h"


namespace efanna2e {

  class IndexMips : public Index {
    public:
      explicit IndexMips(const size_t dimension, const size_t n, Metric m,
                        Index *initializer);

      virtual ~IndexMips();

      virtual void Save(const char *filename) override;
      virtual void Load(const char *filename) override;
      virtual void Search(const float *query, const float *x, size_t k,
                          const Parameters &parameters, unsigned *indices) override;
      virtual void Build(size_t n, const float *data,
                        const Parameters &parameters) override;

      int Search_Mips_IP_Cal(const float *query, const float *x, size_t k,
                          const Parameters &parameters, unsigned *indices, std::vector<int> &init_nodes);

      int Search_Mips_IP_Cal_with_No_SN(const float *query, const float *x, size_t k, const Parameters &parameters, unsigned *indices);

      void init_eps(std::vector<int> &nodes) {
        eps_.resize(nodes.size());
        for (int i = 0; i < nodes.size(); i++) {
          eps_[i] = nodes[i];
        }
      }

      int check_connected_component();

      void SaveData(float *data) {
        data_ = data;
      }

    protected:
      typedef std::vector<std::vector<unsigned>> CompactGraph;
      typedef std::vector<SimpleNeighbors> LockGraph;
      typedef std::vector<nhood> KNNGraph;

      CompactGraph final_graph_;
      Index *initializer_;

      void init_graph(const Parameters &parameters);
      void get_neighbors(const float *query, const Parameters &parameter,
                        std::vector<Neighbor> &retset,
                        std::vector<Neighbor> &fullset);
      void get_neighbors(const unsigned q, const Parameters &parameter,
                        std::vector<Neighbor> &pool, boost::dynamic_bitset<> &flags);
      void sync_prune(unsigned q, std::vector<Neighbor> &pool,
                      const Parameters &parameter, float threshold,
                      SimpleNeighbor *cut_graph_);

      void Link(const Parameters &parameters, SimpleNeighbor *cut_graph_);

      void InterInsert(unsigned n, unsigned range, float threshold,
                      std::vector<std::mutex> &locks, SimpleNeighbor *cut_graph_);

      void Load_nn_graph(const char *filename);

      void strong_connect(const Parameters &parameter);

      void get_refine_neighbors(const unsigned ep, const float *query, const Parameters &parameter,
                        std::vector<Neighbor> &retset,
                        std::vector<Neighbor> &fullset);

      void add_mips_neighbors(const unsigned n, const Parameters &parameter,
                        std::vector<IpNeighbor> &retset,
                        SimpleNeighbor *cut_graph_);
      void get_mips_neighbors(const unsigned ep, const float *query, const Parameters &parameter,
                        std::vector<IpNeighbor> &retset);

      void DFS(boost::dynamic_bitset<> &flag,
              std::vector<std::pair<unsigned, unsigned>> &edges, unsigned root,
              unsigned &cnt);
      bool check_edge(unsigned u, unsigned t);
      void findroot(boost::dynamic_bitset<> &flag, unsigned &root,
                    const Parameters &parameter);
      void DFS_expand(const Parameters &parameter);

    private:
      unsigned width;
      unsigned ep_; //not in use
      std::vector<unsigned> eps_;
      std::vector<std::mutex> locks;
      char *opt_graph_;
      size_t node_size;
      size_t data_len;
      size_t neighbor_len;
      KNNGraph nnd_graph;
      std::vector<float> norms_;
      std::vector<int> init_points_;
  };

}

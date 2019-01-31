#include "../include/union_find.h"


int find_root(std::unordered_map<int, std::pair<int, int>>& parent, int x){
    if(parent[x].first != x){
        parent[x].first = find_root(parent, parent[x].first);
    }
    return parent[x].first;
}


void union_sets(std::unordered_map<int, std::pair<int, int>>& parent, int x, int y){
    int root1 = find_root(parent, x);
    int root2 = find_root(parent, y);
    if (root1 == root2){
        return;
    }
    if(parent[root1].second > parent[root2].second){
        parent[root2].first = root1;
        parent[root2].second += parent[root1].second;
    }
    else {
        parent[root1].first = root2;
        parent[root1].second += parent[root2].second;
    }
}


void make_set(std::unordered_map<int, std::pair<int, int>>& parent, int x){
    parent[x].first = x;
    parent[x].second = 1;

}

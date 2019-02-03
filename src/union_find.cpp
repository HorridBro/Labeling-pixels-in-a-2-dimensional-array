#include "../include/union_find.h"


int find_root(std::unordered_map<int, std::pair<int, int>>& parent, int x){
    if(parent[x].first != x){
        parent[x].first = find_root(parent, parent[x].first);
    }
    return parent[x].first;
}


void union_sets(std::unordered_map<int, std::pair<int, int>>& parent, int x, int y){
    if(parent.find(x) == parent.end()){
        return;
    }
    if(parent.find(y) == parent.end()){
        return;
    }

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


int find_root_tile(int* parent, int* size, int x){
    if(parent[x] != x){
        parent[x] = find_root_tile(parent, size, parent[x]);
    }
    return parent[x];

}



void union_sets_tile(int *parent, int* size, int x , int y){
    int root1 = find_root_tile(parent, size, x);
    int root2 = find_root_tile(parent, size, y);
    if (root1 == root2){
        return;
    }
    if(size[root1] > size[root2]){
        parent[root2] = root1;
        size[root2] += size[root1];
    }
    else {
        parent[root1] = root2;
        size[root1] += size[root2];
    }
}



void make_set_tile(int* parent, int* size, int x){
    parent[x] = x;
    size[x] = 1;
}

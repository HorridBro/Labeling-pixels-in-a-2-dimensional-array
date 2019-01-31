
#ifndef LABELING_PIXELS_IN_A_2_DIMENSIONAL_ARRAY_UNION_FIND_H
#define LABELING_PIXELS_IN_A_2_DIMENSIONAL_ARRAY_UNION_FIND_H
#include "includes.h"
int find_root(std::unordered_map<int, std::pair<int, int>>& parent, int x);
void union_sets(std::unordered_map<int, std::pair<int, int>>& parent, int x, int y);
void make_set(std::unordered_map<int, std::pair<int, int>>& parent, int x);


#endif //LABELING_PIXELS_IN_A_2_DIMENSIONAL_ARRAY_UNION_FIND_H

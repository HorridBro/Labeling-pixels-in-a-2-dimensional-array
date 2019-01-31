#include "../include/utils.h"

// check all 8 neighbours
//vector<pair<int, int>> directions {
//                                    {-1, -1}, {-1, 0}, {-1, 1},
//                                    {1, 1}, {1, 0}, {1, -1},
//                                    {0, -1}, {0, 1}
//                                  };

//check only w, n-w, n ,e neighbours

std::vector<std::pair<int, int>> directions = { {0, -1}, {-1, -1}, {-1, 0}, {-1, 1} };
std::vector<int> colors = {
                                    0x00FF00,
                                    0x0000FF,
                                    0xFF0000,
                                    0x01FFFE,
                                    0xFFA6FE,
                                    0xFFDB66,
                                    0x006401,
                                    0x010067,
                                    0x95003A,
                                    0x007DB5,
                                    0xFF00F6,
                                    0xFFEEE8,
                                    0x774D00,
                                    0x90FB92,
                                    0x0076FF,
                                    0xD5FF00,
                                    0xFF937E,
                                    0x6A826C,
                                    0xFF029D,
                                    0xFE8900,
                                    0x7A4782,
                                    0x7E2DD2,
                                    0x85A900,
                                    0xFF0056,
                                    0xA42400,
                                    0x00AE7E,
                                    0x683D3B,
                                    0xBDC6FF,
                                    0x263400,
                                    0xBDD393,
                                    0x00B917,
                                    0x9E008E,
                                    0x001544,
                                    0xC28C9F,
                                    0xFF74A3,
                                    0x01D0FF,
                                    0x004754,
                                    0xE56FFE,
                                    0x788231,
                                    0x0E4CA1,
                                    0x91D0CB,
                                    0xBE9970,
                                    0x968AE8,
                                    0xBB8800,
                                    0x43002C,
                                    0xDEFF74,
                                    0x00FFC6,
                                    0xFFE502,
                                    0x620E00,
                                    0x008F9C,
                                    0x98FF52,
                                    0x7544B1,
                                    0xB500FF,
                                    0x00FF78,
                                    0xFF6E41,
                                    0x005F39,
                                    0x6B6882,
                                    0x5FAD4E,
                                    0xA75740,
                                    0xA5FFD2,
                                    0xFFB167,
                                    0x009BFF,
                                    0xE85EBE,
                                };

void delete_matrix(int** data, int rows){
    for (int i = 0; i < rows; ++i){
        delete [] data[i];
    }
    delete [] data;
}



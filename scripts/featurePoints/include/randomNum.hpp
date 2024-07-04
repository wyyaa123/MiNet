#ifndef _RANDOMNUM_
#define _RANDOMNUM_ 1
#include <random>
template <typename _Ty>
_Ty getRandomNum (_Ty min, _Ty max) {
    std::random_device rd;  // 随机设备，用于生成随机种子
    std::mt19937 gen(rd()); // 梅森旋转算法，生成随机数序列
    std::uniform_int_distribution<> dis(min, max); // 均匀分布，范围为min到max
    return dis(gen);
}
#endif
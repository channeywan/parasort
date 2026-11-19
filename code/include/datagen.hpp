// 确定性输入生成接口。
// 对应 DESIGN.md 第 4.2：数据生成 节。
// 实验配置与校验均沿用公共框架，算法默认参数保持不变。
#pragma once
#include "common.hpp"
namespace parasort {
uint64_t mix64(uint64_t);
Key generate_key(const Config &, uint64_t global_index);
void generate(Key *, uint64_t count, uint64_t global_start, const Config &, bool serial = false);
void first_touch(Key *, uint64_t, int threads, bool serial = false);
} // namespace parasort

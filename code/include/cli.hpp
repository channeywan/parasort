// 命令行解析接口。
// 对应 DESIGN.md 第 4.1：统一参数 节。
// 实验配置与校验均沿用公共框架，算法默认参数保持不变。
#pragma once
#include "common.hpp"
namespace parasort {
Config parse_cli(int, char **, const std::string &);
std::string help_text();
uint64_t parse_size(const std::string &);
} // namespace parasort

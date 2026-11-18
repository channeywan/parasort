#pragma once
#include "common.hpp"
namespace parasort { Config parse_cli(int,char**,const std::string&); std::string help_text(); uint64_t parse_size(const std::string&); }

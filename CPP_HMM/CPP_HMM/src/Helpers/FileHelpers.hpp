#ifndef FILEHELPERS
#define FILEHELPERS

//#include "stdafx.h"

#include <Windows.h>
#include <vector>
#include <string>

namespace Helper
{
	std::vector<std::string> get_all_files_names_within_folder(std::string folder);

	std::vector<std::string> getFilesPathWithinFolder(std::string folder, bool recursive = true, std::string extension = "");

}

#endif
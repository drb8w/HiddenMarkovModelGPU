//#include "stdafx.h"

#include "FileHelpers.hpp"

#include "StringHelpers.hpp"

#include <stdio.h>
#include <wchar.h>

#include <string>
#include <ctype.h>
#include <algorithm>
#include <functional>
#include <iostream>

using namespace std;

namespace Helper
{

	//wchar_t *convertCharArrayToLPCWSTR(const char* charArray)
	//{
	//	wchar_t* wString = new wchar_t[4096];
	//	MultiByteToWideChar(CP_ACP, 0, charArray, -1, wString, 4096);
	//	return wString;
	//}

	CHAR wide_to_narrow(WCHAR w)
	{
		// simple typecast
		// works because UNICODE incorporates ASCII into itself
		return CHAR(w);
	}

	vector<string> get_all_files_names_within_folder(string folder)
	{
		vector<string> names;
		char search_path[200];
		sprintf(search_path, "%s*.*", folder.c_str());
		WIN32_FIND_DATA fd;
		wchar_t *w_search_path = convertCharArrayToLPCWSTR(search_path);
		HANDLE hFind = ::FindFirstFile(w_search_path, &fd);
		if (hFind != INVALID_HANDLE_VALUE) {
			do {
				// read all (real) files in current folder
				// , delete '!' read other 2 default folder . and ..
				if (!(fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY))
				{
					CHAR dest[MAX_PATH];
					std::transform(fd.cFileName, fd.cFileName + MAX_PATH, dest, wide_to_narrow);
					names.push_back(dest);
				}
			} while (::FindNextFile(hFind, &fd));
			::FindClose(hFind);
		}
		return names;
	}

	vector<string> getFilesPathWithinFolder(string folder, bool recursive, string extension)
	{
		vector<string> names;
		char search_path[200];
		sprintf(search_path, "%s*.*", folder.c_str());
		WIN32_FIND_DATA fd;
		wchar_t *w_search_path = convertCharArrayToLPCWSTR(search_path);
		HANDLE hFind = ::FindFirstFile(w_search_path, &fd);
		if (hFind != INVALID_HANDLE_VALUE) {
			do {
				// read all (real) files in current folder
				// , delete '!' read other 2 default folder . and ..
				if (!(fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY))
				{
					CHAR dest[MAX_PATH];
					std::transform(fd.cFileName, fd.cFileName + MAX_PATH, dest, wide_to_narrow);

					bool append = true;
					if (strcmp(extension.c_str(), "") != 0){
						string destStr(dest);
						size_t iIndex = destStr.rfind(extension);
						if (iIndex < 0 || iIndex > destStr.size())
							append = false;
					}
					if (append)
					{
						string fileName = folder + dest;
						names.push_back(fileName);
					}
				}
				else if (recursive)
				{
					CHAR dest[MAX_PATH];
					std::transform(fd.cFileName, fd.cFileName + MAX_PATH, dest, wide_to_narrow);
					if (strcmp(dest, ".") && strcmp(dest, ".."))
					{
						string subFolder = folder + dest + "/";
						// recursive call
						vector<string> subNames = getFilesPathWithinFolder(subFolder, recursive, extension);
						names.insert(names.end(), subNames.begin(), subNames.end());
					}
				}
			} while (::FindNextFile(hFind, &fd));
			::FindClose(hFind);
		}
		return names;
	}

}
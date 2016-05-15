#ifndef STRINGHELPERS
#define STRINGHELPERS

//#include "stdafx.h"

#include <vector>
#include <string>

namespace Helper
{
	std::vector<std::string> splitString(std::string s, std::string delimiter);

	char *replace(char *st, char *orig, char *repl);

	std::string ArgumentString(int argc, char *argv[], int argNo, std::string &str);

	std::string ArgumentPath(int argc, char *argv[], int argNo, std::string &path);

	int str_hash(const std::string &key, int tableSize = USHRT_MAX);

	std::string ExecutionPath();

	std::string RelativeFileName(std::string fileName, bool withExtension = false);

	std::string CleanString(std::string str, bool numbers = true, bool specialCharacters = true);

	wchar_t *convertCharArrayToLPCWSTR(const char* charArray);

	std::string itos(int i);

}

#endif
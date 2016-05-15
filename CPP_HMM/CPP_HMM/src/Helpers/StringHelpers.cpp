//#include "stdafx.h"
#include "StringHelpers.hpp"

#include <iostream>
#include <Windows.h>

#include <locale>
#include <string>
#include <sstream>

#include <algorithm>

using namespace std;

namespace Helper
{

	vector<string> splitString(string s, string delimiter)
	{
		vector<string> tokens;
		size_t pos = 0;
		std::string token;
		while ((pos = s.find(delimiter)) != std::string::npos) {
			token = s.substr(0, pos);
			tokens.push_back(token);
			s.erase(0, pos + delimiter.length());
		}
		tokens.push_back(s);
		return tokens;
	}

	char *replace(char *st, char *orig, char *repl)
	{
		static char buffer[4096];
		char *ch;
		if (!(ch = strstr(st, orig)))
			return st;
		strncpy(buffer, st, ch - st);
		buffer[ch - st] = 0;
		sprintf(buffer + (ch - st), "%s%s", repl, ch + strlen(orig));
		return buffer;
	}

	std::string ArgumentString(int argc, char *argv[], int argNo, std::string &str)
	{
		if (argc > argNo)
			str = argv[argNo];

		return str;
	}

	wstring widen(const string& str)
	{
		wostringstream wstm;
		const ctype<wchar_t>& ctfacet =
			use_facet< ctype<wchar_t> >(wstm.getloc());
		for (size_t i = 0; i < str.size(); ++i)
			wstm << ctfacet.widen(str[i]);
		return wstm.str();
	}

	string narrow(const wstring& str)
	{
		ostringstream stm;
		const ctype<char>& ctfacet =
			use_facet< ctype<char> >(stm.getloc());
		for (size_t i = 0; i < str.size(); ++i)
			stm << ctfacet.narrow(str[i], 0);
		return stm.str();
	}

	std::string ExecutionPath()
	{
		wchar_t buffer[MAX_PATH];
		GetModuleFileName(NULL, buffer, MAX_PATH);

		// convert to string
		string execPath = narrow(buffer);

		for (int i = 0; i < 100; i++)
			execPath = replace((char *)(execPath.c_str()), "\\", "/");

		// find last slash
		std::size_t found = execPath.find_last_of("/");
		execPath = execPath.substr(0, found + 1);

		return execPath;
	}

	std::string ArgumentPath(int argc, char *argv[], int argNo, std::string &path)
	{
		if (argc > argNo)
			path = argv[argNo];

		for (int i = 0; i < 100; i++)
			path = replace((char *)(path.c_str()), "\\", "/");

		if (strcmp((const char *)&(path[path.size() - 1]), "/"))
			path = path + "/";

		if (path[0] == '.')
		{
			// convert to string
			string execPath = ExecutionPath();
			// concat right parts
			path = execPath + path.substr(2, path.size() - 2);
		}

		return path;
	}

	int str_hash(const string &key, int tableSize) {
		int hashVal = 0;

		for (int i = 0; i < key.length(); i++)
			hashVal = 37 * hashVal + key[i];

		hashVal %= tableSize;

		if (hashVal < 0)
			hashVal += tableSize;

		return hashVal;
	}

	// compact fileName to relative path without extension
	std::string RelativeFileName(std::string fileName, bool withExtension)
	{
		// truncate ExecutionPath();
		std::string execPath = ExecutionPath();

		std::string relativeFileName = fileName;

		std::string::size_type p = fileName.find(execPath);
		if (p != std::string::npos)
		{
			int pos = p + execPath.length() - 1;
			int len = fileName.length() - pos;
			relativeFileName = fileName.substr(pos, len);
		}

		if (!withExtension)
		{
			std::string::size_type p2 = relativeFileName.rfind('.');
			if (p2 != std::string::npos)
			{
				int pos2 = p2;
				relativeFileName = relativeFileName.substr(0, pos2);
			}
		}

		relativeFileName = '.' + relativeFileName;

		return relativeFileName;
	}

	std::string CleanString(std::string str, bool numbers, bool specialCharacters)
	{
		if (numbers)
		{
			char chars[] = "0123456789";
			for (unsigned int i = 0; i < strlen(chars); ++i)
				str.erase(std::remove(str.begin(), str.end(), chars[i]), str.end());
		}

		if (specialCharacters)
		{
			char chars[] = "_";
			for (unsigned int i = 0; i < strlen(chars); ++i)
				str.erase(std::remove(str.begin(), str.end(), chars[i]), str.end());
		}

		return str;
	}

	wchar_t *convertCharArrayToLPCWSTR(const char* charArray)
	{
		wchar_t* wString = new wchar_t[4096];
		MultiByteToWideChar(CP_ACP, 0, charArray, -1, wString, 4096);
		return wString;
	}

	std::string itos(int i)
	{
		char iStr[33];
		_itoa(i, iStr, 10);
		std::string s(iStr);
		return s;
	}
}
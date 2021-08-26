# 获取静态数组和动态数组的长度

总结:
**$\textcolor{blue}{获取char类型字符串数组长度：可以用strlen( )函数}$**

**$\textcolor{blue}{获得string类型的字符串长度：可用string.size()函数}$**

**$\textcolor{blue}{获取数组长度: 使用sizeof(array) / sizeof(array[0])，注意在C/C++中并没有提供直接获取}$**

**$\textcolor{blue}{数组长度的函数vector可以直接由a.size()获得动态数组长度}$**

[TOC]

## 1. 字符串数组

获取字节型的字符串数组长度：可以用strlen( )函数

获得string类型的字符串长度：可用string.size()函数

```cpp
#include <iostream>
#include <string>
using namespace std;
int main(){
	char str1[] = "hello world!"; // 长度为12
    int len_str1 = strlen(str1);
    cout << "str1:" << str1 << endl;
    cout << "str1_len:" << len_str1 << endl;

    string str2 = "hello world"; // 长度为11
    cout << "str2:" << str2 << endl;
    cout << "str2_len:" << str2.size() << endl;
}
```

```markdown
输出：

str1:hello world!
str1_len:12
str2:hello world
str2_len:11
```

## 2. 静态数组

获取数组长度: 使用sizeof(array) / sizeof(array[0])，注意在C/C++中并没有提供直接获取数组长度的函数

```cpp
#include <iostream>
#include <string>
using namespace std;
int main(){
	int b1[] = { 0, 2, 4, 5, 1, 3, 8, 6 };
	cout << "len_b1: " << sizeof(b1)/sizeof(b1[0]) << endl;
}
```

```
输出：
len_b1: 8
```

## 3. 动态数组

数组长度的函数vector可以直接由a.size()获得动态数组长度

```cpp
#include <iostream>
#include <string>
using namespace std;
int main(){
	vector<int> a1 = { 0, 2, 4, 5, 1, 3, 8, 6 };
    cout << "len_a1: " << a1.size() << endl;
}
```

```
输出：
len_a1: 8
```


## C++常用头文件

```c++
// 万能头文件
#include<bits/stdc++.h> 

// 基础头文件
#include <stdio.h> // scanf/printf所需头文件
#include <iostream> // cin/cout 所需头文件
```

### C++常见的字符

| 字符 | 意义                                                         | 字符 | 意义                    |
| ---- | ------------------------------------------------------------ | ---- | ----------------------- |
| '\0' | 空字符，表示字符串的结束标志。空字符是空格、制表符（TAB键产生的字符）、<br />换行符（Enter键所产生的字符）和注释的总称。空白符长度为0 | ' '  | 空格字符，空格符长度为1 |
| '\n' | 换行符                                                       |      |                         |

#### '\0'与' '的区别

```cpp
char a[] = "\0";
char b[] = " ";
cout << strlen(a) << endl;   //0
cout << strlen(b) << endl;   //1
char crr[] = "a b";   //输出是a b
char brr[] = "a\0b";  //输出是a   --------->因为遇到'\0'代表结束
cout << strlen(crr) << endl;
cout << strlen(brr) << endl;
```

# cin/get/getline

主要内容：

1、cin用法,  cin.get()

2、cin.getline()用法

3、getline()用法

4、cin.getline() 与 getline() 的区别

## 1 cin

该操作符是根据后面变量的类型读取数据。

输入结束条件  ：遇到Enter、Space、Tab键。

对结束符的处理 ：丢弃缓冲区中使得输入结束的结束符(Enter、Space、Tab)

* 输入一个数字或者字符

  ```c++
  #include <iostream>
  using namespace std;
  int main ()
  {
  int a,b;
  cin>> a >>b;
  cout<< a+b <<endl;
  }
  ```

* 接收字符串，遇“空格”、“TAB”、“回车”就结束

  ```c++
  #include <iostream>
  
  using namespace std;
  
  int main() {
  	char a[20];
  	cin >> a;
  	cout << a << endl;
  }
  // 输入：wu hui lan
  // 输出：wu
  ```

## 2 cin.get()

该函数有三种格式：无参，一参数，二参数
		即cin.get(), cin.get(char ch), cin.get(array_name,  Arsize)

读取字符的情况：

* 输入结束条件：Enter键
  对结束符处理：不丢弃缓冲区中的Enter

cin.get() 与 cin.get(char ch)用于读取字符，他们的使用是相似的，
即：ch=cin.get() 等价于 cin.get(ch)。

读取字符串的情况：

* cin.get(array_name, Arsize)是用来读取字符串的，可以接受空格字符，遇到Enter结束输入，按照长度(Arsize)读取字符, 会丢弃最后的Enter字符。

* get（char &）和get（void）提供不跳过空白的单字符输入功能；

  函数get（char * , int , char）和getline（char * , int , char）在默认情况下读取整行而不是一个单词；

  ```
  char a[23];
  cin.getline(a, 20, '#') // 以’#‘做分隔符
  cin.get(a, 20, '#') // 以’#‘做分隔符
  ```

## 3 cin.getline()

cin.getline() 与 cin.get(array_name, Arsize)的读取方式差不多，以Enter结束，可以接受空格字符。按照长度(Arsize)读取字符, 会丢弃最后的Enter字符。

但是这两个函数是有区别的：

cin.get(array_name,  Arsize)当输入的字符串超长时，不会引起cin函数的错误，后面的cin操作会继续执行，只是直接从缓冲区中取数据。但是cin.getline()当输入超长时，会引起cin函数的错误，后面的cin操作将不再执行。（具体原因将在下一部分"cin的错误处理"中详细介绍）

* 函数：cin.getline(接收字符串的变量, 接收字符个数, **结束字符**)

* 用法:接收一个**字符串**，可以接收**空格**并输出

  ```c++
  //接收5个字符到m中，其中最后一个为空字符'\0'，所以只看到4个字符输出
  #include <iostream>
  
  using namespace std;
  
  int main() {
  	char a[20];
  	cin.getline(a, 5);  
  	cout << a << endl;
  }
  
  // 接收在‘a'之前的字符
  #include <iostream>
  
  using namespace std;
  
  int main() {
  	char m[20];
  	cin.getline(m, 5, 'a');  
  	cout << m << endl;
  }
  ```

## 4 getline()

用法：接收一个**字符串**，可以接收空格并输出，需包含“#include<string>”

```c++
#include <iostream>
#include <string>

using namespace std;

int main() {
	string str;
	getline(cin, str);  
	cout << str << endl;
}
```

## 注意：

1. cin.getline()属于**istream流**，而getline()属于**string流**，是不一样的两个函数

2. 当同时使用cin>>getline()时，需要注意的是，在cin>>输入流完成之后，getline()之前，需要通过下面代码将回车符作为输入流cin以清除缓存，如果不这样做的话，在控制台上就不会出现getline()的输入提示，而直接跳过，因为程序默认地将之前的变量作为输入流。

   ```cpp
   str="\n";
   getline(cin,str);
   ```

## 5 cin/cout/scanf/printf/gets

### 5.1 **cin / cout** 　　

基本说明：

　　cin代表标准输入设备，使用提取运算符 ">>" 从设备键盘取得数据，送到输入流对象cin中，然后送到内存。

　　cin是输入流，cout是输出流，重载了">>"、"<<"运算符，包含在头文件<font color=red>**< iostream>**</font>中，且**需要配合using namespace std或者std::使用**。

　　先把要输出的东西**存入缓冲区**，再输出，导致效率降低，cin是**自动判断**你的**变量类型**，比如一个char数据只能用默认的char方法取数据。

### 5.2 **scanf / printf**

基本说明：

　　scanf是格式化输入，printf是格式化输出，包含在头文件<font color=red>**<stdio.h>**</font>中。

　　因为scanf是**用指针操作**的，**没有类型安全机制**，比如一个char类型数据你就可以用%f获得输入，而不会报错，但在运行时会出现异常。

　　scanf()函数取数据是遇到**回车、空格、TAB**就会停止，如例1，第一个scanf()会取出"Hello"，而"world!"还在缓冲区中，这样第二个scanf会直接取出这些数据，而不会等待从终端输入。

### 5.3 **gets()**

基本说明：

　　gets()函数用来**从标准输入设备（键盘）读取字符串直到换行符'\n'**结束，但**换行符会被丢弃**，然后在末尾添加'\0'字符。包含在头文件<font color=red>**<stdio.h>**</font>中。

　　gets(s)函数与 scanf("%s",&s) 相似，但不完全相同，使用scanf("%s",&s) 函数输入字符串时存在一个问题，就是如果输入了空格会认为字符串结束，空格后的字符将作为下一个输入项处理，但gets()函数将接收输入的整个字符串直到遇到换行为止。

原型：`char* gets(char* buffer);`

**例1**：scanf/printf

```cpp
#include <stdio.h>
int main()
{
	char str1[20], str2[20];
	scanf("%s", str1);
	printf("%s\n", str1);
	scanf("%s", str2);
	printf("%s\n", str2);
	return 0;
}

// 输入：hello world
// 输出：hello
//		world
// 解释：scanf遇到空格会停止，将后面的内容给第二个scanf提取
```

**例2**：cin/cout

```cpp
#include <iostream>
using namespace std;
int main()
{
	char str1[20], str2[20], str3[20];
	cin >> str1;
	cin >> str2;
	cin >> str3;
	cout << str1 << endl;
	cout << str2 << endl;
	cout << str3 << endl;
}
```

输入：hello world wuhuilan
输出：hello
			world
			wuhuilan

**例3：**gets

```CPP
#include <stdio.h>
int main()
{
　　char str1[20], str2[20];
　　gets(str1); 
　　printf("%s\n",str1);  
　　gets(str2); 
　　printf("%s\n",str2); 
　　return 0;
}
```

测试：
Hello world! [输入]
Hello world! [输出]
12345 [输入]
12345 [输出]

**例4**：cin.get()

```cpp
#include <iostream>
using namespace std;
int main ()
{
char a[20];
cin.get(a, 10);
cout<<a<<endl;
return 0;
}
```

测试一：

输入：abc def[Enter]

输出：abc def

【分析】说明该函数输入字符串时可以接受空格。

测试二

输入：1234567890[Enter]

输出：123456789

【分析】输入超长，则按需要的长度取数据。

```cpp
#include <iostream>
using namespace std;
int main ()
{
char ch, a[20];
cin.get(a, 5);
cin>>ch;
cout<<a<<endl;
cout<<(int)ch<<endl;
return 0;
}
```

测试一

输入：12345[Enter]

输出：

​		1234

​		 53

【分析】第一次输入超长，字符串按长度取了"1234"，而'5'仍残留在缓冲区中，所以第二次输入字符没有从键盘读入，而是直接取了'5'，所以打印的ASCII值是53('5'的ASCII值)。

测试二

输入：1234[Enter]

​			a[Enter]

输出：

​		1234

​		 97

【分析】第二次输入有效，说明该函数把第一次输入后的Enter丢弃了！

cin.getline()

```cpp
#include <iostream>
using namespace std;
int main ()
{
char ch, a[20];
cin.getline(a, 5);
cin>>ch;
cout<<a<<endl;
cout<<(int)ch<<endl;
return 0;
}
```

测试输入：

12345[Enter]

输出：

​		1234

​		 -52

【分析】与cin.get(array_name, Arsize)的例程比较会发现，这里的ch并没有读取缓冲区中的5，而是返回了-52，这里其实cin>>ch语句没有执行，是因为cin出错了！

/*
 * @Author: xudawu
 * @Date: 2021-02-06 18:44:01
 */
#include<iostream>
using namespace std;
class Student
{
private:
    int num;//学生序号
    string name;//学生姓名
public:
    Student();
    ~Student();
    void createStudent(int num1,string name1);//创建学生序号和姓名
    void showStudent();//显示学生信息
};

Student::Student()
{
}

Student::~Student()
{
}
void Student ::createStudent(int num1, string name1)//创建学生序号和姓名
{
    num = num1;
    name = name1;
    // cout << "createStudent(int num1, string name1)" << endl;
}
void Student ::showStudent() //显示学生信息
{
    cout << num << " " << name << endl;
}
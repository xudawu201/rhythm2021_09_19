/*
 * @Author: xudawu
 * @Date: 2021-09-13 09:51:13
 */
#include<iostream>
#include<vector>
using namespace std;

class TwoSum2021_09_13
{
private:
    /* data */
public:
    TwoSum2021_09_13(/* args */);
    ~TwoSum2021_09_13();
    vector<int> twoNumber(vector<int> &nums,int target);
};

vector<int> TwoSum2021_09_13::twoNumber(vector<int> &nums, int target)
{
    int intArraySize = nums.size();
    for (int i = 0; i < intArraySize-1; i++)
    {
        for (int j=i+1; j < intArraySize; j++)
        {
            if (nums[i]+nums[j]==target)//两数和等于目标值
            {
                return {i, j}; //返回数组下标值
                //break;
            }
            
        }
        //break;
    }
    return {}; //未找到满足条件数组下标返回空
}

TwoSum2021_09_13::TwoSum2021_09_13(/* args */)
{
    vector<int> intArray = {2, 7, 11, 15};
    int intTarget = 13;
    vector<int> resultArray;
    resultArray={twoNumber(intArray, intTarget)};//返回的数组下标给结果数组赋初值
    if (resultArray.size()>0)
    {
        cout << resultArray[0] << " " << resultArray[1] << endl;
    }
    else
    {
        cout << "not find target array" << endl;
        ;
    }
    
    //cout << "this is AddTwoNUmber2021_09_13()" << endl;
}

TwoSum2021_09_13::~TwoSum2021_09_13()
{

}
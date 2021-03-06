#include <iostream>
#ifndef __FrequentPatternMining__FPGrowth__
#define __FrequentPatternMining__FPGrowth__
#include <iomanip>
#include <bits/stdc++.h>
#include <algorithm>
#include <cmath>
using namespace std;

struct FPTreeNode {
    int count;
    int name; //item name
    unordered_map<int, FPTreeNode *> children;
    FPTreeNode* parent;
    FPTreeNode* link; //next node that share same name
    
    FPTreeNode(int _name,FPTreeNode* _parent,int _count = 1)
    {
        name = _name;
        parent = _parent;
        count = _count;
        children.clear();
        link = NULL;
    }
    ~FPTreeNode()
    {
        unordered_map<int, FPTreeNode*>::iterator it;
        for (it = children.begin(); it != children.end(); it ++) {
            if(it->second!= NULL)
            {
                delete it->second;
            }
        }
    }
};

struct FPHeaderTableNode {
    int name;
    int freq; //total frequency
    FPTreeNode* head; //link to FPTree
    FPTreeNode* end; 
    FPHeaderTableNode(int _name,int _freq = 1,FPTreeNode* _head = NULL)
    {
        name = _name;
        freq = _freq;
        head = _head;
    }
};

//describe transaction
struct FPTransItem {
    int name;
    int count;
    FPTransItem(int _name,int _count)
    {
        name  = _name;
        count = _count;
    }
};
struct FPTrans {
    vector<FPTransItem> items;
    FPTrans(vector<FPTransItem> &_items)
    {
        items = _items;
    }
};

//result
struct FPFreqResult {
    float freq;
    vector<FPTransItem> items;
};
struct tmpitem {  
    float freq;
    vector<int> items;
};
class FPGrowth {
private:
    float minCountSupport;//absulate minimum support count
    float minSupport; //threshold value
    int   itemCount; //count of items that march minSupport
    int transCount = 0; // count of tansactions

    FPTreeNode* root;
    vector<FPTransItem *> *prefix;
    
    //item name & frequency & head
    vector<FPHeaderTableNode *> FPHeaderTable;
    
    //name > headerTable index (hash map), nameIndex[0] store the max freq item
    unordered_map<int, int> nameIndex;
    
    
    //init HeaderTable and name-to-HeaderTable inidex
    void buildHeaderTable(string fileName);
    void buildHeaderTable(vector<FPTrans> &trans);
    void sortHeaderTable();
    
    //init FPTree
    void buildFPTree(string fileName);
    void buildFPTree(vector<FPTrans> &trans);
    
    //insert a tran to a tree
    void insertTran(vector<int> &items);
    void insertTran(vector<FPTransItem*> &items);
    
    //mining
    void miningFPTree(vector<FPFreqResult> &result);
    void printSingleResult(vector<FPFreqResult> &result);

public:
    FPGrowth(float _minSupport);
    ~FPGrowth();
    void initFromFile(string fileName);
    void initFromTrans(vector<FPTrans> &trans);
    
    void output(vector<FPFreqResult> &result);
    void outputToFile(string fileName);
    static bool compare_items(const tmpitem &num1, const tmpitem &num2){
        int i = 0;
        if(num1.items.size() == num2.items.size()){
            while(i!= num2.items.size()-1){
                if(num1.items[i] != num2.items[i]) break;
                else i++;
            }
            return num1.items[i] < num2.items[i];
        }else return num1.items.size() < num2.items.size();
    }
    //FOR DEBUG
    void outputHeaderTable();
    void outputTran(vector<int> items);
    void outputTree();
    void outputPrefix();
    void outputFreq();
};
#endif /* defined(__FrequentPatternMining__FPGrowth__) */

#pragma -mark public method

FPGrowth::FPGrowth(float _minSupport)
{
    //init root node
    root = new FPTreeNode(0,NULL,0); // number items
    //set min support
    minSupport = _minSupport;
}

void FPGrowth::initFromFile(string fileName)
{
    buildHeaderTable(fileName);
    buildFPTree(fileName);
    prefix = NULL; //insure that first mining will creat a new prefix
}

void FPGrowth::outputToFile(string fileName)
{
    ofstream ofile;
    ofile.open(fileName.c_str());

    vector<FPFreqResult> result;
    output(result);
    vector<tmpitem> sortresult;
    vector<FPFreqResult>::iterator it;
    for (it = result.begin(); it != result.end(); it ++) {
        vector<int>itemset;
        struct tmpitem temp;
        vector<FPTransItem>::iterator itemIt;
        for (itemIt = it->items.begin(); itemIt != it->items.end(); itemIt ++) {
            itemset.push_back(itemIt->name);
        }
        temp.items = itemset;
        temp.freq = it->freq;
        sortresult.push_back(temp);
    }
    for (int i=0 ;i<sortresult.size(); i++) {
        sort(sortresult[i].items.begin(), sortresult[i].items.end());
    }
    sort(sortresult.begin(), sortresult.end(), FPGrowth::compare_items);

    for (int i=0 ;i< sortresult.size(); i++) {
        struct tmpitem temp = sortresult[i];
        vector<int> tmp = temp.items;
        for (int j=0 ;j<tmp.size(); j++) {
            //if(j==0);
            //else cout << ",";
            //cout << tmp[j];
            if(j==0);
            else ofile << ",";
            ofile << tmp[j];
        }
        //cout << transCount;
        //cout << (float)temp.freq/(float)transCount << endl;
        double num = (int)((float)temp.freq/(float)transCount * 10000 + 0.5) / (10000 * 1.0);
        //cout<<":"<< setiosflags(ios::fixed)<<setprecision(4)<<num<<endl;
        ofile << ":" << setiosflags(ios::fixed)<<setprecision(4)<<num<< endl; 
    }
    ofile.close();

}

void FPGrowth::output(vector<FPFreqResult> &result)
{
    miningFPTree(result);
}

//any allocated space in trans need caller to delete immediately
void FPGrowth::initFromTrans(vector<FPTrans> &trans)
{
    buildHeaderTable(trans);
    buildFPTree(trans);
}

#pragma -mark private method

void FPGrowth::buildHeaderTable(string fileName)
{
    ifstream ifile;
    ifile.open(fileName.c_str());

    string tran; //get a transaction
    int index = 0; //index in HeaderTable
    vector< set<int> > vect;
    while (getline(ifile , tran)) {
        transCount ++;
        set<int> set;
        stringstream ss(tran);
        for (int i; ss >> i;) {
            set.insert(i);  
            if (ss.peek() == ',')
                ss.ignore();
        }
        vect.push_back(set);
    }
    ifile.close();

    for(int i=0;i<vect.size();i++){
        set<int> s = vect[i];
        set<int>::iterator it = s.begin();
        //fout << remap(*it);
        while(it != s.end()){
            int item = *it;
            if (nameIndex.find(item) != nameIndex.end()) {
                    FPHeaderTable[nameIndex[item]]->freq ++;
            }
            else{ //item doesn't exit in Header Table
                nameIndex[item] = index;
                FPHeaderTableNode *headerTableNode = new FPHeaderTableNode(item);
                FPHeaderTable.push_back(headerTableNode);
                index ++;
            }   
            it++;
        }
    }

    //delete items that not match min Support
    vector<FPHeaderTableNode *>::iterator it;
    minCountSupport = (float)transCount * minSupport;
    for (it = FPHeaderTable.begin(); it != FPHeaderTable.end(); it++)
    {
        FPHeaderTableNode *node = *it;
        if ((float)(node->freq) < minCountSupport) {
            it = FPHeaderTable.erase(it);
            it --;
        }
    }
    
    sortHeaderTable();
}

void FPGrowth::buildHeaderTable(vector<FPTrans> &trans)
{
    int index = 0; //index in HeaderTable
    vector<FPTrans>::iterator transIt;
    for (transIt = trans.begin(); transIt != trans.end(); transIt++) {
        vector<FPTransItem>::iterator itemIt;
        for (itemIt = transIt->items.begin(); itemIt != transIt->items.end(); itemIt++) {
            if (nameIndex.find((itemIt)->name) != nameIndex.end()) {
                FPHeaderTable[nameIndex[(itemIt)->name]]->freq += (itemIt)->count;
            }
            else{ //item doesn't exit in Header Table
                nameIndex[(itemIt)->name] = index;
                FPHeaderTableNode *headerTableNode = new FPHeaderTableNode((itemIt)->name,(itemIt)->count);
                FPHeaderTable.push_back(headerTableNode);
                index ++;
            }
        }
    }
    
    //delete items that not match min Support
    vector<FPHeaderTableNode *>::iterator it;
    
    for (it = FPHeaderTable.begin(); it != FPHeaderTable.end(); it++)
    {
        FPHeaderTableNode *node = *it;
        if ((float)(node->freq) < minCountSupport) {
            it = FPHeaderTable.erase(it);
            it --;
        }
    }
    
    sortHeaderTable();
}

bool CompareHeaderTableNode(FPHeaderTableNode *a,FPHeaderTableNode *b)
{
    return a->freq > b->freq;
}
void FPGrowth::sortHeaderTable()
{
    sort(FPHeaderTable.begin(), FPHeaderTable.end(), CompareHeaderTableNode);

    //update name index
    int index = 0;
    nameIndex.clear();
    vector<FPHeaderTableNode *>::iterator it;
    for (it = FPHeaderTable.begin(); it != FPHeaderTable.end(); it++)
    {
        FPHeaderTableNode *node = *it;
        nameIndex[node->name] = index;
        index ++;
    }
    itemCount = index;
}


void FPGrowth::insertTran(vector<int> &items)
{
    FPTreeNode *currentTreeNode = root;
    unordered_map<int, FPTreeNode*> *currentChildren;
    currentChildren = &(currentTreeNode->children);
    vector<int>::iterator it;
    for (it = items.begin(); it != items.end(); it ++) {
        
        //if prefix match
        if (currentChildren->find(*it) != currentChildren->end()) {
            (*currentChildren)[*it]->count ++;
            currentTreeNode = (*currentChildren)[*it];
            currentChildren = &(currentTreeNode->children);
        }
        else{
            FPTreeNode *newTreeNode = new FPTreeNode(*it,currentTreeNode);
            (*currentChildren)[*it] = newTreeNode;
            newTreeNode->parent = currentTreeNode; //set parent
            currentTreeNode = newTreeNode;
            currentChildren = &(newTreeNode->children);
            
            //update FPHeaderTable
            if (FPHeaderTable[nameIndex[*it]]->head == NULL) {
                FPHeaderTable[nameIndex[*it]]->head = currentTreeNode;
                FPHeaderTable[nameIndex[*it]]->end = currentTreeNode;
            }
            else {
                FPHeaderTable[nameIndex[*it]]->end->link = currentTreeNode;
                FPHeaderTable[nameIndex[*it]]->end = currentTreeNode;
            }
        }
        

    }
}

void FPGrowth::insertTran(vector<FPTransItem*> &items)
{
    FPTreeNode *currentTreeNode = root;
    unordered_map<int, FPTreeNode*> *currentChildren;
    currentChildren = &(currentTreeNode->children); //a map from item name to children tree node
    
    vector<FPTransItem*>::iterator it;
    for (it = items.begin(); it != items.end(); it ++) {
        
        //if prefix match, search next
        if (currentChildren->find((*it)->name) != currentChildren->end()) {
            
            (*currentChildren)[(*it)->name]->count += (*it)->count;
            
            currentTreeNode = (*currentChildren)[(*it)->name];
            currentChildren = &(currentTreeNode->children);
        }
        else{ //prefix not match
            FPTreeNode *newTreeNode = new FPTreeNode((*it)->name,currentTreeNode);
            
            newTreeNode->count = (*it)->count;
            
            (*currentChildren)[(*it)->name] = newTreeNode;
            newTreeNode->parent = currentTreeNode; //set parent
            currentTreeNode = newTreeNode;
            currentChildren = &(newTreeNode->children);
            
            //update FPHeaderTable
            if (FPHeaderTable[nameIndex[(*it)->name]]->head == NULL) {
                FPHeaderTable[nameIndex[(*it)->name]]->head    = currentTreeNode;
                FPHeaderTable[nameIndex[(*it)->name]]->end     = currentTreeNode;
            }
            else {
                FPHeaderTable[nameIndex[(*it)->name]]->end->link   = currentTreeNode;
                FPHeaderTable[nameIndex[(*it)->name]]->end         = currentTreeNode;
            }
        }

    }
}

//sort a transaction use nameIndex
unordered_map<int, int> *gNameIndex = NULL;
bool CompareItem(int a,int b)
{
    return (*gNameIndex)[a] < (*gNameIndex)[b];
}
bool CompareTransItem(FPTransItem *a,FPTransItem *b)
{
    return (*gNameIndex)[a->name] < (*gNameIndex)[b->name];
}
void FPGrowth::buildFPTree(string fileName)
{
    ifstream ifile;
    ifile.open(fileName.c_str());
    
    string tran; //get a transaction
    vector< set<int> > vect;
    while (getline(ifile , tran)) {
        set<int> set;
        stringstream ss(tran);
        for (int i; ss >> i;) {
            set.insert(i);  
            if (ss.peek() == ',')
                ss.ignore();
        }
        vect.push_back(set);
    }
    ifile.close();

    for(int i=0;i<vect.size();i++){
        vector<int> items; // transaction items
        set<int> s = vect[i];
        set<int>::iterator it = s.begin();
        while(it != s.end()){
            int item = *it;
            if (nameIndex.find(item) != nameIndex.end()) {
                items.push_back(item); //only add item that match minSupport
            }
            it++;
        }
        //sort items
        gNameIndex = &nameIndex;
        sort(items.begin(), items.end(), CompareItem);
        gNameIndex = NULL;
        
        insertTran(items);//insert a modified transaction to FPTree
        
    }
}

void FPGrowth::buildFPTree(vector<FPTrans> &trans)
{
    vector<FPTrans>::iterator transIt;
    vector<FPTransItem>::iterator itemIt;
    for (transIt = trans.begin(); transIt != trans.end(); transIt++) {
        vector<FPTransItem*> items;
        for (itemIt = transIt->items.begin(); itemIt != transIt->items.end(); itemIt++) {
            if (nameIndex.find((itemIt)->name) != nameIndex.end()) {
                items.push_back(&(*itemIt)); //only add item that match minSupport
            }
        }
        //sort items
        gNameIndex = &nameIndex;
        sort(items.begin(), items.end(), CompareTransItem);
        gNameIndex = NULL;
        insertTran(items);//insert a modified transaction to FPTree
    }
}

#pragma -mark  Mining
void FPGrowth::printSingleResult(vector<FPFreqResult> &result)
{
    FPFreqResult resultLine;
    vector<FPTransItem *>::iterator it;
    it = prefix->end() - 1;
    int supportCount = (*it)->count;
    for (; it >= prefix->begin(); it --) {
        FPTransItem item((*it)->name,0);
        resultLine.items.push_back(item);
    }
    resultLine.freq = supportCount;
    result.push_back(resultLine);
}

void FPGrowth::miningFPTree(vector<FPFreqResult> &result)
{
    if (prefix == NULL){ //First mining
        prefix = new vector<FPTransItem *>;
    }
    
    if (FPHeaderTable.size() == 0) { //mining arrives root
        if (prefix->size() > 0) {
            printSingleResult(result);
        }
        return;
    }
    
    if (prefix->size() > 0) { //print current prefix
        printSingleResult(result);
    }
    
    //mining each item in HeaderTable
    vector<FPHeaderTableNode *>::iterator it;
    
    for (it = FPHeaderTable.end() - 1; it >= FPHeaderTable.begin(); it--)
    {
        FPTransItem *newPrefix = new FPTransItem((*it)->name,0);
        vector<FPTrans> trans;
        FPTreeNode *cur;
        for (cur = (*it)->head; cur != (*it)->end; cur = cur->link) {
            vector<FPTransItem> items;
            //search from node to root,build a modified trans
            FPTreeNode *curToRoot;
            int basicCount = cur->count;
            for (curToRoot = cur->parent;curToRoot->parent != NULL;curToRoot = curToRoot->parent){
                FPTransItem item(curToRoot->name,basicCount);
                items.push_back(item);
            }
            FPTrans tranLine = FPTrans(items);
            trans.push_back(tranLine);
        }
        
        //when cur == (*it)->end
        vector<FPTransItem> items;
        //search from node to root,build a modified trans
        FPTreeNode *curToRoot;
        int basicCount = cur->count;
        for (curToRoot = cur->parent;curToRoot->parent != NULL;curToRoot = curToRoot->parent){
            FPTransItem item(curToRoot->name,basicCount);
            items.push_back(item);
        }
        FPTrans tranLine = FPTrans(items);
        trans.push_back(tranLine);
        
        newPrefix->count = (*it)->freq;
        
        //add prefix
        prefix->push_back(newPrefix);
        
        //recursive mining
        FPGrowth *fpTemp = new FPGrowth(minSupport);
        fpTemp->minCountSupport = minCountSupport;
        fpTemp->initFromTrans(trans);
        fpTemp->prefix = prefix;

        fpTemp->miningFPTree(result);
        
        delete fpTemp;
        prefix->pop_back();
    }

}

#pragma -mark destructor
FPGrowth::~FPGrowth()
{
    if (root != NULL) {
        delete root;
    }
    if (prefix != NULL && prefix->size() == 0) {
        vector<FPTransItem *>::iterator it;
        for (it = prefix->begin(); it != prefix->end(); it++)
        {
            if (*it!=NULL) {
                delete *it;
            }
        }
        delete prefix;
    }
    vector<FPHeaderTableNode *>::iterator it;
    for (it = FPHeaderTable.begin(); it != FPHeaderTable.end(); it++)
    {
        delete *it;
    }
}
#pragma -mark debug method

void FPGrowth::outputHeaderTable()
{
    cout << "HeaderTable: "<< endl;
    vector<FPHeaderTableNode *>::iterator it;
    for (it = FPHeaderTable.begin(); it != FPHeaderTable.end(); it++)
    {
        FPHeaderTableNode *node = *it;
        cout << "name:" << node->name
            << "\tfreq: "<< node->freq
            << "\thead: "<<node->head
            << "\tend: "<<node->end
            <<endl;
    }
    cout << endl;
}

void FPGrowth::outputTran(vector<int> items)
{
    vector<int>::iterator it;
    cout << "Tran items: " ;
    for (it = items.begin(); it != items.end(); it ++) {
        cout << *it << " ";
    }
    cout << endl;
}

void FPGrowth::outputTree()
{
    FPTreeNode* cur = root;
    queue<FPTreeNode *>q;
    q.push(cur);
    cout << "Current Tree is:" << endl;
    while (!q.empty()) {
        cur = q.front();
        q.pop();
        cout << "Name: " << cur->name << "\tCount: " << cur->count << "\tChildren num: "<< cur->children.size() << endl;
        unordered_map<int, FPTreeNode*>::iterator it;
        for (it = cur->children.begin(); it != cur->children.end(); it ++) {
            q.push((*it).second);
        }
    }
    cout << endl;
}

void FPGrowth::outputPrefix()
{
    cout << "Current Prefix is: "<<endl;
    vector<FPTransItem *>::iterator it;
    it = prefix->end() - 1;
    int supportCount = (*it)->count;
    for (; it >= prefix->begin(); it --) {
        cout << (*it)->name << " ";
    }
    cout << supportCount << endl;
}

void FPGrowth::outputFreq(){
    vector<FPTransItem *>::iterator it;
    it = prefix->end() - 1;
    int supportCount = (*it)->count;
    cout << "================================" << endl;
    for (; it >= prefix->begin(); it --) {
        cout << (*it)->name << " ";
    }
    cout << supportCount << endl;
    cout << "================================" << endl << endl;
}

int main(int argc, const char * argv[])
{
    
    float minSupport;
    string inputfile;
    string outputfile;
    FPGrowth *fp;   
    cout << "Please input min support , input file name and output file name:" << endl;
    cout << "e.g. 0.2 sample.txt results.txt" << endl;
    cin >> minSupport >> inputfile >> outputfile;

    fp = new FPGrowth(minSupport);
    fp->initFromFile(inputfile);
    fp->outputToFile(outputfile);
    delete fp;

}
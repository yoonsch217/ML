#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <cmath>
#include <numeric>

using namespace std;

void validateTrain(vector<vector<double>>& x, vector<int>& y, vector<double>& theta);

void readFile(string train, vector<vector<double>>& train_x, vector<int>& train_y){
	ifstream readTrain;
	readTrain.open(train);
	if(readTrain.is_open()){
		while(!readTrain.eof()){
			string curLine;
			getline(readTrain, curLine);
			vector<double> tmp;
			tmp.push_back(1.0); // For featureMap, put 1 at the front

			stringstream ss(curLine);
			double d;
			for(int i = 0; i < 2; i++){
				ss >> d;
				tmp.push_back(d);
				if(ss.peek() == ',') ss.ignore();
			}

			train_x.push_back(tmp);
			ss >> d;
			train_y.push_back(d);
		}
	}
}

void readTestFile(string test, vector<vector<double>>& test_x){
	ifstream readTest;
	readTest.open(test);
	if(readTest.is_open()){
		while(!readTest.eof()){
			string curLine;
			getline(readTest, curLine);
			vector<double> tmp;
			tmp.push_back(1.0);

			stringstream ss(curLine);
			double d;
			for(int i = 0; i < 2; i++){
				ss >> d;
				tmp.push_back(d);
				if(ss.peek() == ',') ss.ignore();
			}

			test_x.push_back(tmp);
		}
	}
}

 
int featureMap(vector<vector<double>>& v, int degree){
	for(int i = 0; i < v.size(); i++){
		double x = v[i][1];
		double y = v[i][2];
		for(int deg = 2; deg <= degree; deg++){
			for(int j = 0; j <= deg; j++){
				double cur = pow(x, deg-j) * pow(y, j);
				v[i].push_back(cur);
			}
		}

	}
	return v[0].size();
}


double hFunc(vector<double>& x, vector<double> theta){
	double product = inner_product(theta.begin(), theta.end(), x.begin(), 0.0);
	return 1.0/(1.0 + exp(-product));
}

double costFunction(vector<vector<double>>& train_x, vector<int>& train_y, 
					vector<double>& theta, double lambda){
	int m = train_x.size();

	double reg = (lambda*inner_product(theta.begin()+1, theta.end(), theta.begin()+1, 0.0)) / (2*m);
	
	double sum = 0.0;
	for(int i = 0; i < m; i++){
		double h = hFunc(train_x[i], theta);
		sum += (-1*train_y[i]*log(h) - (1-train_y[i])*log(1-h));
	}

	return (sum/(double)(m)) + reg;
}


vector<double> gradient(vector<vector<double>>& train_x, vector<int>& train_y, 
				vector<double>& theta, double lambda){

	///////////////////

	/*
	int m  = train_x.size();
	vector<double> result;

	for(int j = 0; j < theta.size(); j++){
		double sum = 0.0;		
		for(int i = 0; i < m; i++){
			double h = hFunc(train_x[i], theta);
			sum += ((h-train_y[i])*train_x[i][j]);
			double reg = lambda*theta[j]/m;
			if(j == 0) result.push_back(sum/m);
			else result.push_back(sum/m + reg);
		}
	}
	
	return result;

*/
	////////////////
	
	int m = train_x.size();

	vector<double> reg = theta;
	for(int i = 0; i < reg.size(); i++) reg[i] = reg[i]*lambda/m;


	vector<double> sum = train_x[0];
	for(int i = 0; i < sum.size(); i++) sum[i] = sum[i]*(hFunc(train_x[0], theta) - train_y[0]);

	for(int i = 1; i < m; i++){
		vector<double> tmp = train_x[i];
		for(int j = 0; j < tmp.size(); j++) tmp[j] = tmp[j]*(hFunc(train_x[i], theta) - train_y[i]);

		for(int j = 0; j < sum.size(); j++) sum[j] += tmp[j];	
	}
	
	for(int i = 0; i < sum.size(); i++) sum[i] = sum[i]/m + reg[i];
	
	return sum;
}


void trainTheta(vector<vector<double>>& x, vector<int>& y, vector<double>& theta, double lambda, double alpha, int iterations, vector<vector<double>>& theta_history, vector<double>& cost_history){

	int cnt = 1;

	for(int iter = 0; iter < iterations; iter++){
		vector<double> grad = gradient(x, y, theta, lambda);

		for(int i = 0; i < theta.size(); i++) theta[i] -= alpha*grad[i];
		
		theta_history.push_back(theta);
		cost_history.push_back(costFunction(x, y, theta, lambda));

		
		//For Debugging
		/*
		int checkpoint = iterations / 3;
		
		if(iter == 0 || iter%checkpoint == 0){
			cout << cnt << "th check point" << endl;
			cnt++;
			validateTrain(x, y, theta);
		}
		*/

	}
	cout << endl;

}

void computeResult(vector<vector<double>>& x, vector<int>& y, vector<double>& theta){   //empty vector for y
	for(int i = 0; i < x.size(); i++){
		double hValue = hFunc(x[i], theta);
		if(hValue >= 0.5) y.push_back(1);
		else y.push_back(0);
	}
}

void validateTrain(vector<vector<double>>& x, vector<int>& y, vector<double>& theta){
	vector<int> result;
	computeResult(x, result, theta);
	int count = 0;
	for(int i = 0; i < result.size(); i++){
		if(result[i] == y[i]) count++;
	}

	for(int i = 0; i < 100; i+=10){
		cout << "Specific validation for " << i << "th element" << endl;
		cout << "h value is: " << hFunc(x[i], theta) << endl;
		cout << "Computed result vs Answer: " << result[i] << " vs " << y[i] << endl << endl;
	}

	cout << "num of correct results out of " << result.size() << " is " << count << endl;
	cout << "accuracy: " << (double)(count)*100 / (double)(result.size()) << "%\n\n";
}

void showThetaHistory(vector<vector<double>>& theta_history){
	for(int i = 0; i < theta_history.size(); i += 1+theta_history.size()/100){
		for(int j = 0; j < theta_history[0].size(); j++) cout << theta_history[i][j] << " ";
		cout << endl << endl;
	}
	cout << endl << endl;
}

void showCostHistory(vector<double>& cost_history){
	for(int i = 0; i < cost_history.size(); i += 1+cost_history.size()/100){
		cout << cost_history[i] << " ";
	}
	cout << endl;
}

///////////////////////////////////////////////////
//.......................MAIN....................//
///////////////////////////////////////////////////

int main(int argc, char** argv){
	string train, test;
	train = argv[1];  //train.txt test.txt 순서로 입력받는다
	test = argv[2];

	vector<vector<double>> train_x;
	vector<int> train_y;
	readFile(train, train_x, train_y);

	vector<vector<double>> test_x;
	vector<int> test_y, result;
	readTestFile(test, test_x);
	

	int deg_input; int iterations; double alpha; double lambda;
	deg_input = 15; iterations = 500; alpha = 0.3; lambda = 0.3;
	/*
	cout << "degree of featureMap" << endl;
	cin >> deg_input;
	cout << "num of iterations" << endl;
	cin >> iterations;
	cout << "alpha" << endl;
	cin >> alpha;
	cout << "lambda" << endl;
	cin >> lambda;
	*/

	int dimension = featureMap(train_x, deg_input);
	int dimension2 = featureMap(test_x, deg_input);
	if(dimension != dimension2) cout << "ERROR: dimension differecne" << endl;

	vector<double> theta(dimension, -0.1);	
	vector<vector<double>> theta_history;
	vector<double> cost_history;

	trainTheta(train_x, train_y, theta, lambda, alpha, iterations, theta_history, cost_history);

//	validateTrain(train_x, train_y, theta);
//	showThetaHistory(theta_history);
//	showCostHistory(cost_history);
	
	computeResult(test_x, test_y, theta); 

	for(int i = 0; i < test_y.size(); i++) cout << test_y[i] << endl;

	return 0;
	

}

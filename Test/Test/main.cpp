#include "Log.h"

void Incremente(int& value) {
	value++;
}
int main() {
	double b = 0;
	if (b) {
		int a = 2000;
		a++;
		Logchar("Hello word");
	}
	int c = 2;
	int& ref = c;
	Incremente(ref);
	Logvalue(c);
}
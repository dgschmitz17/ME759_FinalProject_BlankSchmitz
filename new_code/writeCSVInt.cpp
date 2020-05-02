// Function to write data to a comma-separated file
// Author: Dylan Schmitz

#include <cstdlib>
#include <iostream>
#include <cstddef>
#include <fstream>

// Provide some namespace shortcuts
using namespace std;

void writeCSVInt(const char *filename, const size_t *data, size_t n_rows,
                 size_t n_cols) {
  ofstream fileOut(filename);

  for (size_t i = 0; i < n_rows; i++) {
    fileOut << data[i];
    if (i < n_rows) {
      fileOut << "\n";
    }
  } // end for

  fileOut.close();
} // end WriteCSV
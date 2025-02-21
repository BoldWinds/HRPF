#pragma once

#include "common/enum.h"
#include "common/runtime.h"
#include "datastructure/arraylist.h"
#include "datastructure/matrix.h"
#include <float.h>
#include <math.h>
#include <memory.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>


/**
 * @brief
 * INITIALIZE THE SQUARE MATRIX WITH RANDOM VALUE;
 */
void initialize(int dim, Matrix *ha, Matrix *hb) {
  // srand48(time(NULL));
  auto &runtime = Runtime::get_instance();
  auto cpu = runtime.get_cpu();
  (ha)->access(cpu, MemAccess::W);
  (hb)->access(cpu, MemAccess::W);
  for (int i = 0; i < dim; ++i) {
    for (int j = 0; j < dim; ++j) {
      *(ha->get_cdata(i, j)) = (_TYPE)(1);
      *(hb->get_cdata(i, j)) = (_TYPE)(1);
    }
  }
  auto gpu = runtime.get_cpu();
  // ha->access(gpu, MemAccess::R);
  // hb->access(gpu, MemAccess::R);
}

void initialize(int dim, Matrix *ha) {
  // srand48(time(NULL));
  auto &runtime = Runtime::get_instance();
  auto cpu = runtime.get_cpu();
  (ha)->access(cpu, MemAccess::W);
  for (int i = 0; i < dim; ++i) {
    for (int j = 0; j < dim; ++j) {
      *(ha->get_cdata(i, j)) = (_TYPE)(rand() % 100);
    }
  }
}

void initialize(ArrayList *data, int length) {
  srand48(time(NULL));
  auto &runtime = Runtime::get_instance();
  auto cpu = runtime.get_cpu();
  (data)->access(cpu, MemAccess::W);
  for (int i = 0; i < length; ++i) {
    *(data->get_cdata(i)) = (_TYPE)(rand() % 100);
  }
}

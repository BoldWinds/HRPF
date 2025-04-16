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
#include <random>


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
  _TYPE *a = ha->get_cdata();
  size_t ld = ha->get_ld(); 
  std::mt19937 rng(time(0));
  std::uniform_int_distribution<int> dist(0, 10000);
  for (int j = 0; j < dim; ++j) {
    _TYPE *col = a + j * ld; // 指向第 j 列的起始位置
    for (int i = 0; i < dim; ++i) {
      col[i] = (_TYPE)(dist(rng));
    }
  }
}

void initialize(ArrayList *data, int length) {
  srand48(time(NULL));
  auto &runtime = Runtime::get_instance();
  auto cpu = runtime.get_cpu();
  (data)->access(cpu, MemAccess::W);
  _TYPE *a = data->get_cdata();
  std::mt19937 rng(time(0));
  std::uniform_int_distribution<int> dist(0, 10000);
  for (int i = 0; i < length; ++i) {
    a[i] = (_TYPE)(dist(rng));
  }
}

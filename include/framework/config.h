#pragma once

#ifndef ASNC
#define ASNC  1
#endif

#ifndef STREAM_NUM_
#define STREAM_NUM_  10
#endif

#ifndef HANDLE_NUM_
#define HANDLE_NUM_  0
#endif

#ifndef c_num
#define c_num  0
#endif

#ifndef g_num
#define g_num  10
#endif

// #ifndef PARALLEL_FOR
//仅循环需要置为1
#define PARALLEL_FOR 1
// #endif

#define _REDUCE 0
#define T_SIZE (( c_num ) + ( g_num ))
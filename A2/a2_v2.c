#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <math.h>
#include <stdbool.h>
#include <memory.h>
#include <omp.h> 
#include <time.h>
//////////////////////////////////////////////////////////////////////////////////////
// CONSTANTS 
//////////////////////////////////////////////////////////////////////////////////////
#define NO_DIMS 2 
#define DEFAULT_K 5  
#define TIMESTAMP_NO_ITEMS 9
#define MAX_NUM_NEIGHBOURS 4
#define PERIOD_CHECK 10                     //every 10 seconds base checks
#define TIME_MSG_VALID 30
#define TIME_CHECK_THRESHOLD 0.0001
#define MAX_ITER 15
#define NODE_AVAIL_THRESHOLD 2
#define TIME_LOG_VALID 30

#define FNAME_BASE "base_log.txt"
#define FNAME_PREF_NODE "node_log"

#define TAG_TERMINATE 0 
#define TAG_MSG_NODE_TO_BASE 1
#define TAG_BASE_NODE_NON_IMM_COUNT 2
#define TAG_NON_IMM_NEIGHBOURS 3 
#define TAG_PLEA 4
#define TAG_LOG 5 
#define TAG_PLEA_REPLY 6

#define SHIFT_ROW 1 //vertically 
#define SHIFT_COL 0 //horizontally 

#define UPPER_VAL_PROB 100
#define LOWER_VAL_PROB 0
#define PROB_PORT_SWITCHES 90 

#define PORT_CHANGE_STATUS_ITERS 2

#define BASE_OFF 1
#define NODES_OFF 0
#define CART_SHIFT 1 
//////////////////////////////////////////////////////////////////////////////////////
// GLOBAL VARIABLES 
//////////////////////////////////////////////////////////////////////////////////////
int nrows, ncols, k;
int rank_base;
int size;
//////////////////////////////////////////////////////////////////////////////////////
// STRUCTS 
//////////////////////////////////////////////////////////////////////////////////////
struct nodeStruct {  //struct that nodes will send base station 
    int tag;
    int time_stamp[TIMESTAMP_NO_ITEMS];          //reported time  != received time at base station 
    int avail;                                   //
    int num_msgs_neighbours;                     //
    int num_neighbours_avail;                    //
    double comm_time[MAX_NUM_NEIGHBOURS];        //
    int neighbour_avail[MAX_NUM_NEIGHBOURS];     //
    int neighbour_exists[MAX_NUM_NEIGHBOURS];    //top, bottom, left, right  
};
struct baseThreadStruct {
    int pid;
    FILE* pFile;
    struct tm** timestamp_2d_list;
    struct nodeStruct* log_1d_struct_list;
    int* log_1d_flag_list;
    int** avail_2d_list;
    int* nearby_1d_flag_list;
    int** nearby_2d_list;
    MPI_Comm comm_world; 
    int nrows;
    int ncols;
    int k;
    pthread_mutex_t* ptr_mutex_log; 
    pthread_mutex_t* ptr_mutex_run_iter;
    pthread_mutex_t* ptr_mutex_terminate_all;
    pthread_mutex_t* ptr_mutex_grid_avail;
    pthread_mutex_t* ptr_mutex_nearby;
    pthread_mutex_t* ptr_mutex_iter;
    int* ptr_run_iter;
    int* ptr_terminate_all;
    MPI_Datatype valueType;
    int* ptr_iter;
};
struct nodeThreadStruct {
    int pid;
    FILE* pFile;
    int* ptr_avail;
    MPI_Datatype  Valuetype;
    MPI_Comm comm2d;
    MPI_Comm comm_world;
    MPI_Comm comm_group;
    int* ptr_port_avail_list_counter;
    struct tm* shared_table_timestamp_1d_list;
    int* shared_table_1d_list;
    int*  ptr_termination;                   
    int* ptr_run_iter;     
    int* ptr_iter;
    pthread_mutex_t* ptr_mutex_iter;            
    pthread_mutex_t* ptr_mutex_shared_list;  
    pthread_mutex_t* ptr_mutex_run_iter;     
    pthread_mutex_t* ptr_mutex_termination;  
    pthread_mutex_t* ptr_mutex_avail;
};

//////////////////////////////////////////////////////////////////////////////////////
// FUNCTION DECLARATIONS  
//////////////////////////////////////////////////////////////////////////////////////
int leader_io(MPI_Comm comm_world, MPI_Comm comm_group, MPI_Datatype Valuetype);
int follower_io(MPI_Comm comm_world, MPI_Comm comm_group, MPI_Datatype Valuetype);
void store_node_info(int rank, struct nodeStruct value, struct tm** timestamp_2d_list, int** avail_2d_list);
void* baseThreadFunction(void *pArg);
void* nodeThreadFunction(void *pArg);
void log_node_file_shared_list(FILE* pFile, pthread_mutex_t* ptr_mutex_shared_list, struct tm* shared_table_timestamp_1d_list, int* shared_table_1d_list, int* ptr_port_avail_list_counter);
void store_node_info(int rank, struct nodeStruct value, struct tm** timestamp_2d_list, int** avail_2d_list);
void rank_to_coord(int* coord, int grid_rank, int ncols, int zero_one_index);
int coord_to_rank(int* coord, int zero_one_index);
int coord_rc_to_rank(int row, int col, int nrows, int ncols, int zero_one_index);
int get_non_immediate_neighbours(int* non_imm_neigh_list, int* coord, struct tm** timestamp_2d_list, int** avail_2d_list);
struct tm convert_timelist_to_tm(int* time_list);
void convert_tm_to_timelist(int* time_list, struct tm my_time);
int manhat_dist(int x1, int y1, int x2, int y2);
int check_bounds(int row, int col);
int check_bounds_coord(int* coord);
int top_exists(int rank_node);
int bottom_exists(int rank_node);
int left_exists(int rank_node);
int right_exists(int rank_node);
void log_report_to_file(FILE* pFile, int rank, int iter, struct nodeStruct value, int nearby_count, int* nearby_nodes_list, int** avail_2d_list, pthread_mutex_t* mutex_nearby, pthread_mutex_t* mutex_grid_avail);
void print_visual_log(FILE* pFile, int** avail_2d_list);
double calc_time_taken(struct timespec start_time);
void get_time_for_nodeStruct(int* time_stamp);

int main(int argc, char *argv[]) {

    //set the time to Australian/Sydney time 
    setenv("TZ", "Australia/Sydney", 1);
    tzset();

    struct nodeStruct values;
    int dims[NO_DIMS];
    int size, my_rank;
    //start up initial MPI environment 
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    rank_base = size -1;
    //command line arguments for nrows, ncols, availability in that order 
    if (argc == 4) {
        //argv[0] is the program name itself
        nrows = atoi(argv[1]);
        ncols = atoi(argv[2]);
        k = atoi(argv[3]);

        // dims[0] = nrows;
        // dims[1] = ncols;
        
        //check that the number of nodes matches the size (number of mpi processes)
        if((nrows*ncols) != size-1) { //it's size - 1 because we need 1 mpi process for base station
            if(my_rank == 0) printf("ERROR: nrows*ncols = %d * %d = %d != %d\n", nrows, ncols, nrows*ncols, size -1);
            MPI_Finalize();
            return 0;
        }
        //check if availability is at least 1 
        if(k < 1) {
            printf("ERROR: k = %d < 1d\n", k);
            MPI_Finalize();
            return 0;
        }
    }
    //if we don't get any arguments for nrows, ncols, availability in command line, assume that k = 5
    else {
        //it must be so that size = n*m + 1
        nrows = ncols = (int) sqrt(size -1);
        // dims[0] = dims[1] = nrows;
        k = DEFAULT_K;
    } 

    //create new datatype MPI
    MPI_Datatype Valuetype;
    MPI_Datatype type[8] = {MPI_INT,      //tag 
                            MPI_INT,      //timestamp
                            MPI_INT,      //avail
                            MPI_INT,      //num_msgs_neighbours 
                            MPI_INT,      //num_neighbours_avail
                            MPI_DOUBLE,   //comm_time
                            MPI_INT,   //neighbour_avail
                            MPI_INT      //neighbour_exists
    };

    int blockLen[8] = {1, TIMESTAMP_NO_ITEMS, 1, 1, 1, MAX_NUM_NEIGHBOURS, MAX_NUM_NEIGHBOURS, MAX_NUM_NEIGHBOURS}; 

    MPI_Aint disp[8];

    MPI_Get_address(&values.tag, &disp[0]);
    MPI_Get_address(&values.time_stamp, &disp[1]);
    MPI_Get_address(&values.avail, &disp[2]);
    MPI_Get_address(&values.num_msgs_neighbours, &disp[3]);
    MPI_Get_address(&values.num_neighbours_avail, &disp[4]);
    MPI_Get_address(&values.comm_time, &disp[5]);
    MPI_Get_address(&values.neighbour_avail, &disp[6]);
    MPI_Get_address(&values.neighbour_exists, &disp[7]);

    //make relative 
    disp[7] = disp[7] - disp[6];
    disp[6] = disp[6] - disp[5];
    disp[5] = disp[5] - disp[4];
    disp[4] = disp[4] - disp[3];
    disp[3] = disp[3] - disp[2];
    disp[2] = disp[2] - disp[1];
    disp[1] = disp[1] - disp[0];
    disp[0] = 0;  

    // Create MPI struct
    MPI_Type_create_struct(8, blockLen, disp, type, &Valuetype);
    MPI_Type_commit(&Valuetype);

    // decide what group current process is in
    int is_basestation = (my_rank == rank_base) ? 1 : 0; // Decide which processes are in the separate group
    
    //create a group for the grid and a gruop for the baseStation
    MPI_Comm comm_group; // newcomm will the communicator specific to each group 
    MPI_Comm_split(MPI_COMM_WORLD, is_basestation, 1, &comm_group); // each group will use this to communicate with its group members  

    if(!is_basestation) {
        follower_io(MPI_COMM_WORLD, comm_group, Valuetype);
    }
    else {
        leader_io(MPI_COMM_WORLD, comm_group, Valuetype);
    }
    
    MPI_Type_free(&Valuetype);
    MPI_Finalize();
    return 0;
}

void* baseThreadFunction(void *pArg) {
    struct baseThreadStruct *args = (struct baseThreadStruct *)pArg;

    int pid                                    = args->pid;
    FILE* p_log                                = args->pFile;
    struct tm** timestamp_2d_list              = args->timestamp_2d_list;
    struct nodeStruct* log_1d_struct_list      = args->log_1d_struct_list;
    int* log_1d_flag_list                      = args->log_1d_flag_list;
    int** avail_2d_list                        = args->avail_2d_list;
    int* nearby_1d_flag_list                   = args->nearby_1d_flag_list;
    int** nearby_2d_list                       = args->nearby_2d_list;
    MPI_Comm comm_world                        = args->comm_world; 
    int nrows                                  = args->nrows;
    int ncols                                  = args->ncols; 
    int k                                      = args->k;
    pthread_mutex_t* ptr_mutex_log             = args->ptr_mutex_log; 
    pthread_mutex_t* ptr_mutex_run_iter        = args->ptr_mutex_run_iter;
    pthread_mutex_t* ptr_mutex_terminate_all   = args->ptr_mutex_terminate_all;
    pthread_mutex_t* ptr_mutex_grid_avail      = args->ptr_mutex_grid_avail;
    pthread_mutex_t* ptr_mutex_nearby          = args->ptr_mutex_nearby;
    pthread_mutex_t* ptr_mutex_iter            = args->ptr_mutex_iter;
    int* ptr_run_iter                          = args->ptr_run_iter;
    int* ptr_terminate_all                     = args->ptr_terminate_all;
    MPI_Datatype valueType                     = args->valueType;
    int* ptr_iter                              = args->ptr_iter;

    int coord[2];
    struct nodeStruct value;

    int size;
    MPI_Comm_size(comm_world, &size);

    struct timespec start_time, curr_time;
    double time_taken = 0.0;

    int non_imm_neigh_count; 
    int non_imm_list_len = 2*(nrows+ncols);
    int non_imm_neigh_list[non_imm_list_len];

    int request_temp; 
    MPI_Status* status_list = (MPI_Status*)calloc(size-1, sizeof(MPI_Status)); //set all the statuses to MPI_Empty 
    int* flag_list = (int*)calloc(size-1, sizeof(int));

    int termination_msg = 1;
    int temp_run_iter;
    int temp_terminate_all; 
    int temp_iter;

    if(pid == 0) {
        clock_gettime(CLOCK_MONOTONIC, &start_time); 
        pthread_mutex_lock(ptr_mutex_iter);
        if(!BASE_OFF) {printf("[MUTEX [ITER] - LOCKED]\n");}
        *ptr_iter = 1;
        pthread_mutex_unlock(ptr_mutex_iter);
        if(!BASE_OFF) {printf("[MUTEX [ITER] - UNLOCKED]\n");}

        while(*ptr_iter < MAX_ITER + 1) {
            time_taken = calc_time_taken(start_time);
            
        
            if(fmod(time_taken, (double) PERIOD_CHECK) >= TIME_CHECK_THRESHOLD) {
                if(!BASE_OFF) {printf("[MUTEX [run-iter] - LOCKED]\n");}
                pthread_mutex_lock(ptr_mutex_run_iter);
                *ptr_run_iter = 0;
                pthread_mutex_unlock(ptr_mutex_run_iter);
                if(!BASE_OFF) {printf("[MUTEX [run_iter] - UNLOCKED]\n");}
                continue;
            }
            if(!BASE_OFF) {printf("--------------------------------------------------------[ITER] base: %d-------------------------------------------\n\n",   *ptr_iter);}
            if(!BASE_OFF) {printf("========================================================================================COMMMUNICATION-------------------------------------------\n\n");}
        
            pthread_mutex_lock(ptr_mutex_run_iter);
            if(!BASE_OFF) {printf("[MUTEX [run_iter] - LOCKED]\n");}
            *ptr_run_iter = 1; 
            pthread_mutex_unlock(ptr_mutex_run_iter);
            if(!BASE_OFF) {printf("[MUTEX [run_iter] - UNLOCKED]\n");}

            for(int n = 0; n < size - 1; n ++) {
                // probe for any messages from this node
                MPI_Iprobe(n, TAG_MSG_NODE_TO_BASE, MPI_COMM_WORLD, &flag_list[n], &status_list[n]);
                if(flag_list[n]) {
                    flag_list[n] = 0;//set back to zero 
                    rank_to_coord(coord, n, ncols, 0);
                    MPI_Recv(&value, 1, valueType, n, TAG_MSG_NODE_TO_BASE, comm_world, &status_list[n]);

                    if(!BASE_OFF) {printf("----------------------------------------------------------------------------report received >>>> rank = %d   coord = (%d, %d)<<<<<<<<\n -------------\n", n, coord[0],coord[1]);}
                    
                    pthread_mutex_lock(ptr_mutex_grid_avail);
                    if(!BASE_OFF) {printf("[MUTEX [grid_avail] - LOCKED]\n");}
                    store_node_info(n, value, timestamp_2d_list, avail_2d_list);
                    pthread_mutex_unlock(ptr_mutex_grid_avail);
                    if(!BASE_OFF) {printf("[MUTEX [grid_avail] - UNLOCKED]\n");}

                    //logger
                    pthread_mutex_lock(ptr_mutex_log);
                    if(!BASE_OFF) {printf("[MUTEX [log] - LOCKED]\n");}
                    log_1d_struct_list[n] = value;
                    log_1d_flag_list[n] = 1;
                    pthread_mutex_unlock(ptr_mutex_log);
                    if(!BASE_OFF) {printf("[MUTEX [log] - UNLOCKED]\n");}

                    //check for plea
                    if(value.tag == TAG_PLEA) {
                        
                        pthread_mutex_lock(ptr_mutex_grid_avail);
                        if(!BASE_OFF) {printf("[MUTEX [grid_avail] - LOCKED]\n");}
                        non_imm_neigh_count = get_non_immediate_neighbours(non_imm_neigh_list, coord, timestamp_2d_list, avail_2d_list);
                        pthread_mutex_unlock(ptr_mutex_grid_avail);
                        if(!BASE_OFF) {printf("[MUTEX [grid_avail] - UNLOCKED]\n");}

                        //send the no of non-imm neighbours to reporting plea node 
                        MPI_Isend(&non_imm_neigh_count, 1, MPI_INT, n, TAG_BASE_NODE_NON_IMM_COUNT, comm_world, &request_temp);

                        pthread_mutex_lock(ptr_mutex_nearby);
                        if(!BASE_OFF) {printf("[MUTEX [nearby] - LOCKED]\n");}
                        nearby_1d_flag_list[n] = non_imm_neigh_count;
                        
                        //if non-imm neigh count > 0, send the list to the reporting node 
                        if(non_imm_neigh_count > 0) {
                            MPI_Isend(&non_imm_neigh_list, non_imm_list_len, MPI_INT, n, TAG_NON_IMM_NEIGHBOURS, comm_world, &request_temp);
                            nearby_2d_list[n] = non_imm_neigh_list;
                        }
                        pthread_mutex_unlock(ptr_mutex_nearby);
                        if(!BASE_OFF) {printf("[MUTEX [nearby] - UNLOCKED]\n");}
                    }
        
                }
            }
            pthread_mutex_lock(ptr_mutex_iter);
            if(!BASE_OFF) {printf("[MUTEX [iter] - LOCKED]\n");}
            *ptr_iter = *ptr_iter + 1;
            pthread_mutex_unlock(ptr_mutex_iter);
            if(!BASE_OFF) {printf("[MUTEX [iter] - UNLOCKED]\n");}

        }
        //mutex - terminate all
        pthread_mutex_lock(ptr_mutex_terminate_all);
        if(!BASE_OFF) {printf("[MUTEX [terminate_all] - LOCKED]\n");}
        *ptr_terminate_all = 1;
        pthread_mutex_unlock(ptr_mutex_terminate_all);
        if(!BASE_OFF) {printf("[MUTEX [terminate_all] - UNLOCKED]\n");}
        
        //send termination message to all other nodes in the grid 
        for(int i = 0; i < size-1;i ++) {
            MPI_Isend(&termination_msg, 1, MPI_INT, i, TAG_TERMINATE, comm_world, &request_temp);
        }
        if(!BASE_OFF) {printf("----------------------------------Sent termination out to all nodes -----------------------------------------\n");}

    }
    else {
        pthread_mutex_lock(ptr_mutex_terminate_all);
        if(!BASE_OFF) {printf("[MUTEX [terminate_all] - LOCKED]\n");}
        temp_terminate_all = *ptr_terminate_all;
        pthread_mutex_unlock(ptr_mutex_terminate_all);
        if(!BASE_OFF) {printf("[MUTEX [terminate_all] - UNLOCKED]\n");}


        while(!temp_terminate_all) {
            if(!BASE_OFF) {printf("========================================================================================LOGGING-------------------------------------------\n\n");}
            if(!BASE_OFF) {printf("[MUTEX [run_iter] - LOCKED]\n");}
            pthread_mutex_lock(ptr_mutex_run_iter);
            temp_run_iter = *ptr_run_iter;
            pthread_mutex_unlock(ptr_mutex_run_iter);
            if(!BASE_OFF) {printf("[MUTEX [run_iter] - UNLOCKED]\n");}

            if(!temp_run_iter) {
                continue;
            }
        
            for(int n = 0; n > size - 1; n++) {

                //check if there is a flag 
                if(log_1d_flag_list[n]) {
                    //mutex - using shared data between the threads at base 
                    pthread_mutex_lock(ptr_mutex_log);
                    if(!BASE_OFF) {printf("[MUTEX [log] - LOCKED]\n");}
                    log_1d_flag_list[n] = 0; //unset flag
                    value = log_1d_struct_list[n];  
                    pthread_mutex_unlock(ptr_mutex_log);
                    if(!BASE_OFF) {printf("[MUTEX [log] - UNLOCKED]\n");}

                    pthread_mutex_lock(ptr_mutex_iter);
                    if(!BASE_OFF) {printf("[MUTEX [iter] - LOCKED]\n");}
                    temp_iter = *ptr_iter;
                    pthread_mutex_unlock(ptr_mutex_iter);
                    if(!BASE_OFF) {printf("[MUTEX [iter] - UNLOCKED]\n");}

                    log_report_to_file(p_log, n, temp_iter, value, nearby_1d_flag_list[n], nearby_2d_list[n], avail_2d_list, ptr_mutex_nearby, ptr_mutex_grid_avail); 
    
                }
            }
            pthread_mutex_lock(ptr_mutex_terminate_all);
            if(!BASE_OFF) {printf("[MUTEX [terminate_all] - LOCKED]\n");}
            temp_terminate_all = *ptr_terminate_all;
            pthread_mutex_unlock(ptr_mutex_terminate_all);
            if(!BASE_OFF) {printf("[MUTEX [terminate_all] - UNLOCKED]\n");}
        }
        if(!BASE_OFF) {printf("--------------------------------------------------------------------------------------------------LOGGING EXITED---------------------------------------------------------------------\n");}
       
    }
    return NULL;
}

int leader_io(MPI_Comm comm_world, MPI_Comm comm_group, MPI_Datatype Valuetype) {

    int num_threads = 2;
    pthread_t tid[num_threads];
	int threadNum[num_threads];
    int size;
    MPI_Comm_size(comm_world, &size);

    //2d array to store the timestamp (in tm) when the last msg was reported (not logged) from a node 
    struct tm** timestamp_2d_list = (struct tm**)calloc(nrows, sizeof( struct tm*));
    for(int i = 0; i < nrows; i++) {
        timestamp_2d_list[i] = (struct tm*)calloc(ncols, sizeof( struct tm));
    }

    //intiialise timestap = 
    time_t curr_time;
    time(&curr_time);
    struct tm time_tm;
    time_tm = *localtime(&curr_time);

    for(int i = 0 ; i < nrows; i++) {
        for(int j = 0; j < ncols; j++) {
            timestamp_2d_list[i][j] =  time_tm;
        }
    }

    //2d array to store the node avail value for the last msg received from that node 
    int** avail_2d_list = (int**)calloc(nrows, sizeof(int*));
    for(int i = 0; i < nrows; i++) {
        avail_2d_list[i] = (int*)calloc(ncols, sizeof(int));
    }   

    for(int i = 0 ; i < nrows; i++) {
        for(int j = 0; j < ncols; j++) {
            avail_2d_list[i][j] =  k;
        }
    }

    //1d array to store flags for all nodes 
    int* log_1d_flag_list = (int*)calloc(size-1, sizeof(int));

    //1d array to store received log from a node 
    struct nodeStruct* log_1d_struct_list = (struct nodeStruct*)calloc(size-1, sizeof(struct nodeStruct));

    //1d array to flag if there is a list for nearby neighbours
    int* nearby_1d_flag_list = (int*)calloc(size-1, sizeof(int));

    //2d array to store nearby nodes for each node 
    int nearby_nodes_list_len = 2*(nrows + ncols);
    int** nearby_2d_list = (int**)calloc(size-1, sizeof(int*));
    for(int i = 0; i < size - 1; i++) {
        nearby_2d_list[i] = (int*)calloc(nearby_nodes_list_len, sizeof(int));
    }

    int terminate_all = 0;
    int run_iter = 0;
    int iter;

    FILE* pFile;
    pFile = fopen(FNAME_BASE, "w");

    struct baseThreadStruct baseThreadValue;
    pthread_mutex_t mutex_log, mutex_run_iter, mutex_terminate_all, mutex_nearby, mutex_grid_avail, mutex_iter;

    if (pthread_mutex_init(&mutex_log, NULL) != 0) {
        perror("Mutex initialization failed");
        return 1;
    }
    if (pthread_mutex_init(&mutex_run_iter, NULL) != 0) {
        perror("Mutex initialization failed");
        return 1;
    }
    if (pthread_mutex_init(&mutex_terminate_all, NULL) != 0) {
        perror("Mutex initialization failed");
        return 1;
    }
    if (pthread_mutex_init(&mutex_nearby, NULL) != 0) {
        perror("Mutex initialization failed");
        return 1;
    }
    if (pthread_mutex_init(&mutex_grid_avail, NULL) != 0) {
        perror("Mutex initialization failed");
        return 1;
    }
    if (pthread_mutex_init(&mutex_iter, NULL) != 0) {
        perror("Mutex initialization failed");
        return 1;
    }
    struct baseThreadStruct baseThreadValue_list[num_threads];

    for(int i = 0; i < num_threads;i++){
        baseThreadValue.pid                     = i;
        baseThreadValue.pFile                   = pFile;
        baseThreadValue.timestamp_2d_list       = timestamp_2d_list;
        baseThreadValue.log_1d_struct_list      = log_1d_struct_list;
        baseThreadValue.log_1d_flag_list        = log_1d_flag_list;
        baseThreadValue.avail_2d_list           = avail_2d_list;
        baseThreadValue.nearby_1d_flag_list     = nearby_1d_flag_list;
        baseThreadValue.nearby_2d_list          = nearby_2d_list;
        baseThreadValue.comm_world              = MPI_COMM_WORLD; 
        baseThreadValue.nrows                   = nrows ;
        baseThreadValue.ncols                   = ncols;
        baseThreadValue.k                       = k;
        baseThreadValue.ptr_mutex_log           = &mutex_log; 
        baseThreadValue.ptr_mutex_run_iter      = &mutex_run_iter;
        baseThreadValue.ptr_mutex_terminate_all = &mutex_terminate_all;
        baseThreadValue.ptr_mutex_grid_avail    = &mutex_grid_avail;
        baseThreadValue.ptr_mutex_nearby        = &mutex_nearby;
        baseThreadValue.ptr_mutex_iter          = &mutex_iter;
        baseThreadValue.ptr_run_iter            = &run_iter;
        baseThreadValue.ptr_terminate_all       = &terminate_all;
        baseThreadValue.valueType               = Valuetype;
        baseThreadValue.ptr_iter                = &iter;
        baseThreadValue_list[i]                 = baseThreadValue;
    }

    for (int pid = 0; pid < num_threads; pid++){
		pthread_create(&tid[pid], NULL, baseThreadFunction, &baseThreadValue_list[pid]);
	}

	for (int pid = 0; pid < num_threads; pid++){
		pthread_join(tid[pid], NULL);
	}
    if(!BASE_OFF) {printf("----------------------------------------------------------------------------------------------------------BASE DONE--------------------------------------------------\n\n");}

    //destroy all mutexes
    pthread_mutex_destroy(&mutex_log);
    pthread_mutex_destroy(&mutex_run_iter);
    pthread_mutex_destroy(&mutex_terminate_all);
    pthread_mutex_destroy(&mutex_grid_avail);
    pthread_mutex_destroy(&mutex_nearby);
    pthread_mutex_destroy(&mutex_iter);


    //free
    for(int i = 0; i < nrows; i ++) {
        free(timestamp_2d_list[i]);
        free(avail_2d_list[i]);
    }

    for(int i = 0; i < size-1;i++) {
        free(nearby_2d_list[i]);
    }
    free(timestamp_2d_list);
    free(avail_2d_list);
    free(log_1d_flag_list);
    free(log_1d_struct_list);
    free(nearby_1d_flag_list);
    free(nearby_2d_list);

    fclose(pFile);
    pFile = NULL;
    return 0;
}

int follower_io(MPI_Comm comm_world, MPI_Comm comm_group, MPI_Datatype Valuetype) { 
    int num_threads = k + 1;
    pthread_t tid[num_threads];
	int threadNum[num_threads];
    int dims[NO_DIMS];
    int reorder, ierr;
    int nDims = NO_DIMS;
    int coord[NO_DIMS];
    int port_avail = k;

    //create cartesian mapping
    MPI_Comm comm2d;
    int wrap_around[NO_DIMS];
    wrap_around[0] = 0; //no wrapping around in 1st dim
    wrap_around[1] = 0; //no wrapping around in 2nd dim
    reorder = 0;  // so that we can find rank(process in comm2D) = x_coord*nrows + ncols;
    ierr = 0;
    dims[0] = nrows;
    dims[1] = ncols;
    //create cartesian topology for process 
    MPI_Dims_create(nrows*ncols, NO_DIMS, dims); //we want size-1 processes to be in the grid (nrows by ncols) + 1 basestation 
    ierr = MPI_Cart_create(comm_group, //we create this grid from the group communicator 
                    nDims,
                    dims,
                    wrap_around,
                    reorder,
                    &comm2d);
    if(ierr != 0) {
        printf("ERROR[%d] creating CART\n", ierr);
        return 0;
    }
    //setup
    int rank_comm, rank_grid, rank_world;
    MPI_Comm_rank(comm_group, &rank_comm);

    //shared array for all charging nodes of length something 
    struct nodeThreadStruct nodeThreadValue;
    struct nodeThreadStruct nodeThreadValue_list[num_threads];
    pthread_mutex_t mutex_iter, mutex_shared_list, mutex_run_iter, mutex_termination, mutex_avail;

    if (pthread_mutex_init(&mutex_iter, NULL) != 0) {
        perror("Mutex initialization failed");
        return 1;
    }
    if (pthread_mutex_init(&mutex_shared_list, NULL) != 0) {
        perror("Mutex initialization failed");
        return 1;
    }
    if (pthread_mutex_init(&mutex_run_iter, NULL) != 0) {
        perror("Mutex initialization failed");
        return 1;
    }
    if (pthread_mutex_init(&mutex_termination, NULL) != 0) {
        perror("Mutex initialization failed");
        return 1;
    }
    if (pthread_mutex_init(&mutex_avail, NULL) != 0) {
        perror("Mutex initialization failed");
        return 1;
    }

    int size_shared_list = k;
    struct tm* shared_table_timestamp_1d_list = (struct tm*)calloc(size_shared_list, sizeof(struct tm)); 
    int* shared_table_1d_list = (int*)calloc(size_shared_list, sizeof(int)); ;

    MPI_Comm_rank(comm_world, &rank_world);
    MPI_Comm_rank(comm_group, &rank_comm);
    MPI_Comm_rank(comm2d, &rank_grid);
    MPI_Cart_coords(comm2d, rank_grid, nDims, coord);
    char fname_node[80];
    snprintf(fname_node, sizeof(fname_node), "node_logs/node_rank_%d_coord_%d-%d.txt", rank_world, rank_comm, rank_grid, coord[0], coord[1]);
    FILE* pFile;
    pFile = fopen(fname_node, "w");
    int port_avail_list_counter = 0;

    int termination = 0;
    int run_iter = 0;
    int iter =1;
    for(int i = 0; i < num_threads;i++){
        nodeThreadValue.pid                              = i;
        nodeThreadValue.pFile                            = pFile;
        nodeThreadValue.ptr_avail                        = &port_avail;
        nodeThreadValue.Valuetype                        = Valuetype; 
        nodeThreadValue.comm2d                           = comm2d;
        nodeThreadValue.comm_world                       = comm_world;
        nodeThreadValue.comm_group                       = comm_group;
        nodeThreadValue.ptr_port_avail_list_counter      = &port_avail_list_counter;
        nodeThreadValue.shared_table_timestamp_1d_list   = shared_table_timestamp_1d_list;
        nodeThreadValue.shared_table_1d_list             = shared_table_1d_list;
        nodeThreadValue.ptr_termination                  = &termination;
        nodeThreadValue.ptr_run_iter                     = &run_iter;
        nodeThreadValue.ptr_iter                         = &iter;
        nodeThreadValue.ptr_mutex_iter                   = &mutex_iter;
        nodeThreadValue.ptr_mutex_shared_list            = &mutex_shared_list;
        nodeThreadValue.ptr_mutex_run_iter               = &mutex_run_iter;
        nodeThreadValue.ptr_mutex_termination            = &mutex_termination;
        nodeThreadValue.ptr_mutex_avail                  = &mutex_avail;
        nodeThreadValue_list[i]                          = nodeThreadValue;
    }
    if(!NODES_OFF) {printf("--------------------------------------------------------[NODE START] %d (%d, %d)-------------------------------------------\n\n", rank_grid, coord[0], coord[1]);}

    for (int pid = 0; pid < num_threads; pid++){
		pthread_create(&tid[pid], NULL, nodeThreadFunction, &nodeThreadValue_list[pid]);
	}

	for (int pid = 0; pid < num_threads; pid++){
		pthread_join(tid[pid], NULL);
	}
    
    fclose(pFile);
    //destroy mutexes
    pthread_mutex_destroy(&mutex_iter);
    pthread_mutex_destroy(&mutex_shared_list);
    pthread_mutex_destroy(&mutex_run_iter);
    pthread_mutex_destroy(&mutex_termination);
    pthread_mutex_destroy(&mutex_avail);

    //free - memory management 
    free(shared_table_timestamp_1d_list);
    free(shared_table_1d_list);
    if(!NODES_OFF) {printf("--------------------------------------------------------[NODE DONE] %d (%d, %d)-------------------------------------------\n\n", rank_grid, coord[0], coord[1]);}
}
void* nodeThreadFunction(void *pArg) {

    struct nodeThreadStruct *args = (struct nodeThreadStruct *)pArg;
    int pid                                    = args->pid;
    FILE* pFile                                = args->pFile;
    int* ptr_avail                             = args->ptr_avail;
    MPI_Datatype     Valuetype                 = args->Valuetype;
    MPI_Comm      comm2d                       = args->comm2d;
    MPI_Comm comm_world                        = args->comm_world;
    MPI_Comm comm_group                        = args->comm_group;
    int* ptr_port_avail_list_counter           = args->ptr_port_avail_list_counter;
    struct tm* shared_table_timestamp_1d_list  = args->shared_table_timestamp_1d_list;
    int*            shared_table_1d_list       = args->shared_table_1d_list;
    int*  ptr_termination                      = args->ptr_termination;
    int* ptr_run_iter                          = args->ptr_run_iter;
    int* ptr_iter                              = args->ptr_iter;
    pthread_mutex_t* ptr_mutex_iter            = args->ptr_mutex_iter;
    pthread_mutex_t* ptr_mutex_shared_list     = args->ptr_mutex_shared_list;
    pthread_mutex_t* ptr_mutex_run_iter        = args->ptr_mutex_run_iter;
    pthread_mutex_t* ptr_mutex_termination     = args->ptr_mutex_termination;
    pthread_mutex_t* ptr_mutex_avail           = args->ptr_mutex_avail;

    int termination_msg = 0;
    struct timespec start_time, curr_time;
    double time_taken = 0.0;
    int temp_port_avail;

    MPI_Request send_request[4];
    MPI_Request receive_request[4];

    int nbr_i_lo, nbr_i_hi; //for top and bottom  
    int nbr_j_lo, nbr_j_hi; //for left and right 
    int disp = 1;
    int coord[2];
    int rank_grid;
    int size;
    MPI_Comm_size(comm_world, &size);
    int rank_base = size-1;
    MPI_Comm_rank(comm2d, &rank_grid);
    //get row neighbours: top and bottom neighbours (if they exist)
    MPI_Cart_shift(comm2d, SHIFT_ROW, CART_SHIFT, &nbr_i_lo, &nbr_i_hi);  //lo = top, hi = bot
    //get col neighbours: left and right neighbours (if they exist)
    MPI_Cart_shift(comm2d, SHIFT_COL, CART_SHIFT, &nbr_j_lo, &nbr_j_hi); //lo = left, hi = right
    MPI_Cart_coords(comm2d, rank_grid, NO_DIMS, &coord);

    int row = row = coord[0]; int col = coord[1];

    int recv_plea_top = -1; int recv_plea_bot = -1; int recv_plea_left= -1; int recv_plea_right = -1;
    int recv_reply_top = -1; int recv_reply_bot = -1; int recv_reply_left= -1; int recv_reply_right = -1;

    struct nodeStruct node_value;
    int neighbour_exists[MAX_NUM_NEIGHBOURS], top_exists, bottom_exists, left_exists, right_exists;
    neighbour_exists[0] = top_exists = check_bounds(row-1, col); neighbour_exists[1] = bottom_exists = check_bounds(row+1, col); neighbour_exists[2] = left_exists = check_bounds(row, col-1) ; neighbour_exists[3] = right_exists = check_bounds(row, col+1);
    int num_neighbours = 0;
    for(int i = 0;i <MAX_NUM_NEIGHBOURS; i++) { num_neighbours += neighbour_exists[i];}
    for(int i = 0 ; i < 4; i++) {node_value.neighbour_exists[i] = neighbour_exists[i];}
    
    int flag_list[4];
    for(int i=0; i< MAX_NUM_NEIGHBOURS; i++) {flag_list[i] = 0;}
    int plea_msg = 1;
    
    int expecting_reply_from_neighbours[MAX_NUM_NEIGHBOURS]; 
    for(int i = 0; i > MAX_NUM_NEIGHBOURS; i++) {expecting_reply_from_neighbours[i] = 0;}
    int flag_reply_from_neigh_list[4];
    for(int i=0; i< MAX_NUM_NEIGHBOURS; i++) {flag_reply_from_neigh_list[i] = 0;}
    int flag_terminate = 0;

    struct timespec start_time_neigh_list[MAX_NUM_NEIGHBOURS];
    for(int i = 0; i < MAX_NUM_NEIGHBOURS; i++) {clock_gettime(CLOCK_MONOTONIC, &start_time_neigh_list[i]);}

    int time_stamp[TIMESTAMP_NO_ITEMS];

    int top_avail, bot_avail, left_avail, right_avail;
    int num_msgs_neighbours = 0; int num_neighbours_avail;
    int expecting_reply_from_base = 0;
    int non_imm_neigh_list_len = 2*(nrows + ncols);
    int non_imm_neigh_list[non_imm_neigh_list_len];
    int non_imm_neigh_count = -1;

    int iter = 1;
    int temp_run_iter;
    int port_in_use_new; int port_in_use = 0;
    MPI_Status temp_status;
    int flag_base_non_imm_count = 0; int flag_non_imm_neigh_list = 0;
    MPI_Status status_list[4];
    int service;
    int expecting_reply_from_any_neighbours = 0;

    if(pid == k) {
        clock_gettime(CLOCK_MONOTONIC, &start_time); 
        pthread_mutex_lock(ptr_mutex_iter);
        if(!NODES_OFF) {printf("[MUTEX [ITER] - LOCKED]\n");}
        *ptr_iter = 1;
        pthread_mutex_unlock(ptr_mutex_iter);
        if(!NODES_OFF) {printf("[MUTEX [ITER] - UNLOCKED]\n");}
        
        while(!termination_msg) {
            //check for termination
            MPI_Iprobe(rank_base, TAG_TERMINATE, comm_world, &flag_terminate, &temp_status);
            if(flag_terminate) {
                MPI_Recv(&termination_msg, 1, MPI_INT, rank_base, TAG_TERMINATE, comm_world, &temp_status); //receive to prevent buffer overflow 
            }   
            if(!NODES_OFF) {printf("[MUTEX [termination] - LOCKED]\n");}
            pthread_mutex_lock(ptr_mutex_termination);
            *ptr_termination = termination_msg;
            pthread_mutex_unlock(ptr_mutex_termination);
            if(!NODES_OFF) {printf("[MUTEX [termination] - UNLOCKED]\n");}

            time_taken = calc_time_taken(start_time);

            pthread_mutex_lock(ptr_mutex_run_iter);
            if(!NODES_OFF) {printf("[MUTEX [run_iter] - LOCKED]\n");}
            if(fmod(time_taken, (double) PERIOD_CHECK) > TIME_CHECK_THRESHOLD) {
                *ptr_run_iter = 0;
                pthread_mutex_unlock(ptr_mutex_run_iter);
                if(!NODES_OFF) {printf("[MUTEX [run_iter] - UNLOCKED]\n");}
                continue;
            }
            *ptr_run_iter = 1;
            pthread_mutex_unlock(ptr_mutex_run_iter);
            if(!NODES_OFF) {printf("[MUTEX [run_iter] - UNLOCKED]\n");}

            if(!NODES_OFF) {printf("--------------------------------------------------------[ITER] node %d (%d,%d): %d-------------------------------------------\n\n", rank_grid, coord[0], coord[1], iter);}

            //check if ports avail 
            pthread_mutex_lock(ptr_mutex_avail);
            temp_port_avail = *ptr_avail;
            pthread_mutex_unlock(ptr_mutex_avail);

            //probe for replies from base - may not exist 
            MPI_Iprobe(rank_base, TAG_BASE_NODE_NON_IMM_COUNT, comm_world, &flag_base_non_imm_count, &temp_status);
            MPI_Iprobe(rank_base, TAG_NON_IMM_NEIGHBOURS, comm_world, &flag_non_imm_neigh_list, &temp_status);
            if(flag_base_non_imm_count & expecting_reply_from_base) {
                MPI_Recv(&non_imm_neigh_count, 1, MPI_INT, rank_base, TAG_BASE_NODE_NON_IMM_COUNT, comm_world, &temp_status); //receive to prevent buffer overflow 
                flag_base_non_imm_count = 0;
                expecting_reply_from_base =  0;
            }

            if(non_imm_neigh_count > 0 && flag_non_imm_neigh_list) {
                MPI_Recv(&non_imm_neigh_list, non_imm_neigh_list_len, MPI_INT, rank_base, TAG_NON_IMM_NEIGHBOURS, comm_world, &temp_status); //receive to prevent buffer overflow 
                flag_non_imm_neigh_list = 0;
                expecting_reply_from_base = 0;
                //log to file if need be 
            }

            //check for any pleas from neighbours  + check for replies from neighbours for your plea 
            //probe first 
            if(top_exists) {
                MPI_Iprobe(nbr_i_lo, TAG_PLEA, comm2d, &flag_list[0], &status_list[0]);
                MPI_Iprobe(nbr_i_lo, TAG_PLEA_REPLY, comm2d, &flag_reply_from_neigh_list[0], &status_list[0]);
            }
            if(bottom_exists) { 
                MPI_Iprobe(nbr_i_hi, TAG_PLEA, comm2d, &flag_list[1], &status_list[1]);
                MPI_Iprobe(nbr_i_hi, TAG_PLEA_REPLY, comm2d, &flag_reply_from_neigh_list[1], &status_list[1]);
            }
            if(bottom_exists) { 
                MPI_Iprobe(nbr_j_lo, TAG_PLEA, comm2d, &flag_list[2], &status_list[2]);
                MPI_Iprobe(nbr_j_lo, TAG_PLEA_REPLY, comm2d, &flag_reply_from_neigh_list[2], &status_list[2]);
            }
            if(bottom_exists) { 
                MPI_Iprobe(nbr_j_hi, TAG_PLEA, comm2d, &flag_list[3], &status_list[3]);
                MPI_Iprobe(nbr_j_lo, TAG_PLEA_REPLY, comm2d, &flag_reply_from_neigh_list[3], &status_list[3]);
            }

            //receive pleas from neighbours if probed 
            if(top_exists && flag_list[0]) {
                MPI_Recv(&recv_plea_top, 1, MPI_INT, nbr_i_lo, TAG_PLEA, comm2d, &receive_request[0]); //receive to prevent buffer overflow 
                flag_list[0] = 0;
                node_value.neighbour_avail[0] = recv_plea_top;
                MPI_Isend(&temp_port_avail, 1, MPI_INT, nbr_i_lo, TAG_PLEA_REPLY, comm2d, &send_request[0]);
            }
            if(bottom_exists && flag_list[1]) {
                MPI_Recv(&recv_plea_bot, 1, MPI_INT, nbr_i_hi, TAG_PLEA, comm2d, &receive_request[1]); //receive to prevent buffer overflow 
                flag_list[1] = 0;
                node_value.neighbour_avail[1] = recv_plea_bot;
                MPI_Isend(&temp_port_avail, 1, MPI_INT, nbr_i_hi, TAG_PLEA_REPLY, comm2d, &send_request[1]);
            }
            if(left_exists && flag_list[2]) {
                MPI_Recv(&recv_plea_left, 1, MPI_INT, nbr_j_lo, TAG_PLEA, comm2d, &receive_request[2]); //receive to prevent buffer overflow 
                flag_list[2] = 0;
                node_value.neighbour_avail[2] = recv_plea_left;
                MPI_Isend(&temp_port_avail, 1, MPI_INT, nbr_j_lo, TAG_PLEA_REPLY, comm2d, &send_request[2]);
            }   
            if(right_exists && flag_list[3]) {
                MPI_Recv(&recv_plea_right, 1, MPI_INT, nbr_j_hi, TAG_PLEA, comm2d, &receive_request[3]); //receive to prevent buffer overflow 
                flag_list[3] = 0;
                node_value.neighbour_avail[3] = recv_plea_right;
                MPI_Isend(&temp_port_avail, 1, MPI_INT, nbr_j_hi, TAG_PLEA_REPLY, comm2d, &send_request[3]); 
            }

            //check for plea replies
            if(top_exists && expecting_reply_from_neighbours[0] && flag_reply_from_neigh_list[0]) {
                MPI_Recv(&recv_reply_top, 1, MPI_INT, nbr_i_lo, TAG_PLEA_REPLY, comm2d, &receive_request[0]); //receive to prevent buffer overflow 
                flag_reply_from_neigh_list[0] = 0;
                node_value.neighbour_avail[0] = recv_reply_top;
                node_value.comm_time[0] = calc_time_taken(start_time_neigh_list[0]);
            }
            if(bottom_exists && expecting_reply_from_neighbours[1] && flag_reply_from_neigh_list[1]) {
                MPI_Recv(&recv_reply_bot, 1, MPI_INT, nbr_i_hi, TAG_PLEA_REPLY, comm2d, &receive_request[1]); //receive to prevent buffer overflow 
                flag_reply_from_neigh_list[1] = 0;
                node_value.neighbour_avail[1] = recv_reply_bot;
                node_value.comm_time[1] = calc_time_taken(start_time_neigh_list[1]);
            }
            if(left_exists && expecting_reply_from_neighbours[2] && flag_reply_from_neigh_list[2]) {
                MPI_Recv(&recv_reply_left, 1, MPI_INT, nbr_j_lo, TAG_PLEA_REPLY, comm2d, &receive_request[2]); //receive to prevent buffer overflow 
                flag_reply_from_neigh_list[2] = 0;
                node_value.neighbour_avail[2] = recv_reply_left;
                node_value.comm_time[2] = calc_time_taken(start_time_neigh_list[2]);
            }   
            if(right_exists && expecting_reply_from_neighbours[3] && flag_reply_from_neigh_list[3]) {
                MPI_Recv(&recv_reply_right, 1, MPI_INT, nbr_j_hi, TAG_PLEA_REPLY, comm2d, &receive_request[3]); //receive to prevent buffer overflow 
                flag_reply_from_neigh_list[3] = 0;
                node_value.neighbour_avail[3] = recv_reply_right;
                node_value.comm_time[3] = calc_time_taken(start_time_neigh_list[3]);
            } 

            //if the port is at capacity

            if(temp_port_avail < NODE_AVAIL_THRESHOLD) {
                if(!NODES_OFF) {printf("[low ports]: %d\n", temp_port_avail);}
                top_avail = 0;
                if(top_exists) {
                    //received neither a plea nor (!waiting for a reply))
                    if(recv_plea_top == -1 && !expecting_reply_from_neighbours[0]) {
                        MPI_Isend(&temp_port_avail, 1, MPI_INT, nbr_i_lo, TAG_PLEA, comm2d, &send_request[0]);
                        expecting_reply_from_neighbours[0] = 1;
                        num_msgs_neighbours++;
                    }
                    //if expecting a reply and received a reply 
                    if(expecting_reply_from_neighbours[0] && recv_reply_top != -1) {
                        expecting_reply_from_neighbours[0] = 0;
                        top_avail = (recv_plea_top >= NODE_AVAIL_THRESHOLD);
                    }
                }
                bot_avail = 0;
                if(bottom_exists) {
                    //received neither a plea nor (!waiting for a reply))
                    if(recv_plea_bot == -1 && !expecting_reply_from_neighbours[1]) {
                        MPI_Isend(&temp_port_avail, 1, MPI_INT, nbr_i_hi, TAG_PLEA, comm2d, &send_request[1]);
                        expecting_reply_from_neighbours[1] = 1;
                        num_msgs_neighbours++;
                    }
                    //if expecting a reply and received a reply 
                    if(expecting_reply_from_neighbours[1] && recv_reply_bot != -1) {
                        expecting_reply_from_neighbours[1] = 0;
                        bot_avail = (recv_plea_bot >= NODE_AVAIL_THRESHOLD);
                    }
                }
                left_avail = 0;
                if(left_exists) {
                    //received neither a plea nor (!waiting for a reply))
                    if(recv_plea_left == -1 && !expecting_reply_from_neighbours[2]) {
                        MPI_Isend(&temp_port_avail, 1, MPI_INT, nbr_j_lo, TAG_PLEA, comm2d, &send_request[2]);
                        expecting_reply_from_neighbours[2] = 1;
                        num_msgs_neighbours++;
                    }
                    //if expecting a reply and received a reply 
                    if(expecting_reply_from_neighbours[2] && recv_reply_left != -1) {
                        expecting_reply_from_neighbours[2] = 0;
                        left_avail = (recv_plea_left >= NODE_AVAIL_THRESHOLD);
                    }
                }
                right_avail = 0;
                if(right_exists) {
                    //received neither a plea nor (!waiting for a reply))
                    if(recv_plea_right == -1 && !expecting_reply_from_neighbours[3]) {
                        MPI_Isend(&temp_port_avail, 1, MPI_INT, nbr_j_hi, TAG_PLEA, comm2d, &send_request[3]);
                        expecting_reply_from_neighbours[3] = 1;
                        num_msgs_neighbours++;
                    }
                    //if expecting a reply and received a reply 
                    if(expecting_reply_from_neighbours[3] && recv_reply_right != -1) {
                        expecting_reply_from_neighbours[3] = 0;
                        right_avail = (recv_reply_right >= NODE_AVAIL_THRESHOLD);
                    }
                }

                num_neighbours_avail = top_avail + bot_avail + left_avail + right_avail;
                for(int i = 0; i < 4; i ++)  {expecting_reply_from_any_neighbours = expecting_reply_from_any_neighbours || expecting_reply_from_neighbours[i];}

                //PLEA to base: if no neighbours available  + !expecting_msgs_from_any_neighbours 
                if(num_neighbours_avail == 0 && !expecting_reply_from_any_neighbours) {
                    get_time_for_nodeStruct(time_stamp);
                    for(int x = 0; x < TIMESTAMP_NO_ITEMS; x++) { node_value.time_stamp[x] = time_stamp[x];}
                    node_value.avail = temp_port_avail;
                    node_value.num_msgs_neighbours = num_msgs_neighbours;
                    node_value.num_neighbours_avail = num_neighbours_avail;
                    node_value.tag = TAG_PLEA;
                    MPI_Isend(&node_value, 1, Valuetype, rank_base, TAG_MSG_NODE_TO_BASE, comm_world, &temp_status);
                    num_msgs_neighbours = 0; //reset
                    expecting_reply_from_base = 1;
                }
                //REPORT to base: even if there are some neighbours avail, you log 
                if(num_neighbours_avail > 0 && !expecting_reply_from_any_neighbours) {

                    get_time_for_nodeStruct(time_stamp);
                    for(int x = 0; x < TIMESTAMP_NO_ITEMS; x++) { node_value.time_stamp[x] = time_stamp[x];}
                    node_value.avail = temp_port_avail;
                    node_value.num_msgs_neighbours = num_msgs_neighbours;
                    node_value.num_neighbours_avail = num_neighbours_avail;
                    node_value.tag = TAG_LOG;
                    MPI_Isend(&node_value, 1, Valuetype, rank_base, TAG_MSG_NODE_TO_BASE, comm_world, &temp_status);
                    num_msgs_neighbours = 0; //reset
                }
            }
            else {
                for(int i = 0; i < 4; i++) {expecting_reply_from_neighbours[i] = 0;}
            }

            recv_reply_top = -1; recv_reply_bot = -1; recv_reply_left= -1; recv_reply_right = -1;
            recv_plea_top = -1; recv_plea_bot = -1; recv_plea_left= -1; recv_plea_right = -1;
            
            pthread_mutex_lock(ptr_mutex_iter);
            if(!BASE_OFF) {printf("[MUTEX [iter] - LOCKED]\n");}
            *ptr_iter = *ptr_iter + 1;
            pthread_mutex_unlock(ptr_mutex_iter);
            if(!BASE_OFF) {printf("[MUTEX [iter] - UNLOCKED]\n");}
 
        }
    }
    else {
        while(!termination_msg) {
            pthread_mutex_lock(ptr_mutex_iter);
            if(!NODES_OFF) {printf("[MUTEX [ITER] - LOCKED]\n");}
            iter = *ptr_iter;
            pthread_mutex_unlock(ptr_mutex_iter);
            if(!NODES_OFF) {printf("[MUTEX [ITER] - UNLOCKED]\n");}
            
            pthread_mutex_lock(ptr_mutex_termination);
            termination_msg = *ptr_termination;
            pthread_mutex_unlock(ptr_mutex_termination);
            if(termination_msg) {
                break;
            }
            
            pthread_mutex_lock(ptr_run_iter);
            temp_run_iter = *ptr_run_iter;
            pthread_mutex_unlock(ptr_run_iter);

            pthread_mutex_lock(ptr_mutex_avail);
            temp_port_avail = *ptr_avail;
            pthread_mutex_unlock(ptr_mutex_avail);

            if(!temp_run_iter) {
                continue;
            }
            
            port_in_use_new = (int) (((rand() % (UPPER_VAL_PROB - LOWER_VAL_PROB + 1)) + LOWER_VAL_PROB) < PROB_PORT_SWITCHES);
            service = -1;

            //if port not used before, but switches to being used 
            if(!port_in_use && port_in_use_new) {
                service = 0;
            }
            //if port becomes free 
            else if(port_in_use && !port_in_use_new) {
                service = 1;
            }
            //if the availability table full
            if(service != -1) {
                if(!NODES_OFF) {printf("\n\n$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ ------YESSSS\n\n\n");}

                get_time_for_nodeStruct(time_stamp);
            
                pthread_mutex_lock(ptr_mutex_shared_list);
                if(*ptr_port_avail_list_counter > k-1) {
                    for(int j = 1; j < k; j ++) {
                        shared_table_timestamp_1d_list[j-1] = shared_table_timestamp_1d_list[j];
                        shared_table_1d_list[j-1] = shared_table_1d_list[j];
                    }
                    *ptr_port_avail_list_counter--;
                }
                if(service == 0 ){
                    temp_port_avail--;
                }   
                else if (service == 1){
                    temp_port_avail++;
                }
                shared_table_timestamp_1d_list[*ptr_port_avail_list_counter] =  convert_timelist_to_tm(time_stamp);
                shared_table_1d_list[*ptr_port_avail_list_counter] = temp_port_avail;

                *ptr_port_avail_list_counter++;
                pthread_mutex_unlock(ptr_mutex_shared_list);  

                pthread_mutex_lock(ptr_mutex_avail);
                *ptr_avail = temp_port_avail;
                pthread_mutex_unlock(ptr_mutex_avail);

                //log this to a file 
                log_node_file_shared_list(pFile, ptr_mutex_shared_list, shared_table_timestamp_1d_list, shared_table_1d_list, ptr_port_avail_list_counter);
            }
            port_in_use = port_in_use_new;
            
        }
    }
    return NULL;
}

//////////////////////////////////////////////////////////////////////////////////////
// AUXILLIARY FUNCTIONS
//////////////////////////////////////////////////////////////////////////////////////
void log_node_file_shared_list(FILE* pFile, pthread_mutex_t* ptr_mutex_shared_list, struct tm* shared_table_timestamp_1d_list, int* shared_table_1d_list, int* ptr_port_avail_list_counter) {
    pthread_mutex_lock(ptr_mutex_shared_list);
    if(*ptr_port_avail_list_counter == 0) {
        pthread_mutex_unlock(ptr_mutex_shared_list);
        return;
    }
    fprintf(pFile, "------------------------------------------------------------------------------------------------------------\n\n");
    char time_logged_buffer[80];
    for(int i = 0; i < *ptr_port_avail_list_counter; i++) {
        strftime(time_logged_buffer, sizeof(time_logged_buffer), "%a %Y-%m-%d %H:%M:%S", &shared_table_timestamp_1d_list[i]);
        fprintf(pFile, "%s    %d\n", time_logged_buffer, shared_table_1d_list[i]);
    }
    fprintf(pFile, "------------------------------------------------------------------------------------------------------------\n\n");
    pthread_mutex_unlock(ptr_mutex_shared_list);
    return;
}

void store_node_info(int rank, struct nodeStruct value, struct tm** timestamp_2d_list, int** avail_2d_list) {
    struct tm time_tm = convert_timelist_to_tm(value.time_stamp);
    int coord[2];
    rank_to_coord(coord, rank, ncols, 0);
    int coord_temp[2];
    int row_temp, col_temp;
    
    //store info about the node reporting 
    timestamp_2d_list[coord[0]][coord[1]] = time_tm;
    avail_2d_list[coord[0]][coord[1]] = value.avail;

    //store info about neighbours of node reporting 
    if(top_exists(rank)) {  
        row_temp = coord[0]-1;
        col_temp = coord[1];
        timestamp_2d_list[row_temp][col_temp] = time_tm;
        avail_2d_list[row_temp][col_temp] = value.neighbour_avail[0];
    }

    if(bottom_exists(rank)) {
        row_temp = coord[0]+1;
        col_temp = coord[1];
        timestamp_2d_list[row_temp][col_temp] = time_tm;
        avail_2d_list[row_temp][col_temp] = value.neighbour_avail[1];
    }

    if(left_exists(rank)) {
        row_temp = coord[0];
        col_temp = coord[1]-1;
        timestamp_2d_list[row_temp][col_temp] = time_tm;
        avail_2d_list[row_temp][col_temp] = value.neighbour_avail[2];
    }

    if(right_exists(rank)) {
        row_temp = coord[0];
        col_temp = coord[1]+1;
        timestamp_2d_list[row_temp][col_temp] = time_tm;
        avail_2d_list[row_temp][col_temp] = value.neighbour_avail[3];
    }
}


void rank_to_coord(int* coord, int grid_rank, int ncols, int zero_one_index) {
    int row, col;
    row = grid_rank / ncols;
    col = grid_rank % ncols ;
    if(zero_one_index) {
        coord[0] = row;
        coord[1] = col;
    }
    else {
        coord[0] = row + 1;
        coord[1] = col + 1;
    }
}

int coord_to_rank(int* coord, int zero_one_index) {
    int row = coord[0];
    int col = coord[1];
    int rank;
    if(!zero_one_index) {
        rank = ncols * row + col;
    }
    else {
        rank = ncols * (row - 1) + (col -1);
    }
    return rank;
}

int coord_rc_to_rank(int row, int col, int nrows, int ncols, int zero_one_index) {
    int rank;
    if(!zero_one_index) {
        rank = ncols * row + col;
    }
    else {
        rank = ncols * (row - 1) + (col -1);
    }
    return rank;
}

int get_non_immediate_neighbours(int* non_imm_neigh_list, int* coord, struct tm** timestamp_2d_list, int** avail_2d_list) {

    time_t curr_time;
    time_t temp_time;
    double time_diff;
    
    int counter = 0;
    int row = coord[0];
    int col = coord[1];
    int disp = 2;

    int disp_top_left = manhat_dist(0, 0, coord[0], coord[1]);
    int disp_top_right = manhat_dist(0, ncols-1, coord[0], coord[1]);
    int disp_bottom_left = manhat_dist(nrows -1, 0, coord[0], coord[1]);
    int disp_bottom_right = manhat_dist(nrows - 1, ncols-1, coord[0], coord[1]);
    int max_disp = fmax(disp_top_left, fmax(disp_top_right, fmax(disp_bottom_left, disp_bottom_right)));

    row = row + disp;
    col = col;

    time(&curr_time);
    struct tm time_tm;
    time_tm = *localtime(&curr_time);
    while (disp < max_disp) {
        //bottom to left  <- ^
        for(int i = 0; i < disp; i++)
        {
            if(check_bounds(row, col)) {
                temp_time = mktime(&timestamp_2d_list[row][col]); //convert from struct tm to time_t 
                time_diff = difftime(curr_time, temp_time);

                //if the time has expired, assume it is available 
                if (time_diff > TIME_LOG_VALID) {
                    timestamp_2d_list[row][col] = time_tm;
                    avail_2d_list[row][col]  = k; //full capacity
                    non_imm_neigh_list[counter] = coord_rc_to_rank(row, col, nrows, ncols, 0);
                    counter++;
                }
                else if(time_diff < TIME_LOG_VALID && avail_2d_list[row][col] >= NODE_AVAIL_THRESHOLD) {
                    non_imm_neigh_list[counter] = coord_rc_to_rank(row, col, nrows, ncols, 0);
                    counter++;
                }
            }   
            row--;
            col--;
        }
        //left to top
        for(int i = 0; i < disp ; i++)
        {

            if(check_bounds(row, col)) {
                temp_time = mktime(&timestamp_2d_list[row][col]);
                time_diff = difftime(curr_time, temp_time);

                if (time_diff > TIME_LOG_VALID) {
                    timestamp_2d_list[row][col] = time_tm;
                    avail_2d_list[row][col]  = k; //full capacity
                    non_imm_neigh_list[counter] = coord_rc_to_rank(row, col, nrows, ncols, 0);
                    counter++;
                }

                if(time_diff < TIME_LOG_VALID && avail_2d_list[row][col] >= NODE_AVAIL_THRESHOLD) {
                    non_imm_neigh_list[counter] = coord_rc_to_rank(row, col, nrows, ncols, 0);
                    counter++;
                }
            }   
            row--;
            col++;
        }
        //top to right
        for(int i = 0; i < disp; i++)
        {
            if(check_bounds(row, col)) {
                temp_time = mktime(&timestamp_2d_list[row][col]); //convert from struct tm to time_t 
                time_diff = difftime(curr_time, temp_time);
                
                if (time_diff > TIME_LOG_VALID) {
                    timestamp_2d_list[row][col] = time_tm;
                    avail_2d_list[row][col]  = k; //full capacity
                    non_imm_neigh_list[counter] = coord_rc_to_rank(row, col, nrows, ncols, 0);
                    counter++;
                }

                if(time_diff < TIME_LOG_VALID && avail_2d_list[row][col] >= NODE_AVAIL_THRESHOLD) {
                    non_imm_neigh_list[counter] = coord_rc_to_rank(row, col, nrows, ncols, 0);
                    counter++;
                }
            }   
            row++;
            col++;
        }
        //right to bottom 
        for(int i = 0; i < disp + 1; i++)
        {
            if(check_bounds(row, col)) {
                temp_time = mktime(&timestamp_2d_list[row][col]); //convert from struct tm to time_t 
                time_diff = difftime(curr_time, temp_time);

                if (time_diff > TIME_LOG_VALID) {
                    timestamp_2d_list[row][col] = time_tm;
                    avail_2d_list[row][col]  = k; //full capacity
                    non_imm_neigh_list[counter] = coord_rc_to_rank(row, col, nrows, ncols, 0);
                    counter++;
                }
                if(time_diff < TIME_LOG_VALID && avail_2d_list[row][col] >= NODE_AVAIL_THRESHOLD) {
                    non_imm_neigh_list[counter] = coord_rc_to_rank(row, col, nrows, ncols, 0);
                    counter++;
                }
            }   
            row++;
            col--;   
        }
    }
    return counter;
}

struct tm convert_timelist_to_tm(int* time_list) {
    struct tm my_time;
    my_time.tm_sec   =  time_list[0];     // Seconds
    my_time.tm_min   =  time_list[1];    // Minutes
    my_time.tm_hour  =  time_list[2];   // Hour (24-hour format)
    my_time.tm_mday  =  time_list[3];   // Day of the month
    my_time.tm_mon   =  time_list[4];     // Month (0-11, so October is 9)
    my_time.tm_year  =  time_list[5];  // Years since 1900, so 2021 - 1900 = 121
    my_time.tm_wday  =  time_list[6];    // Day of the week (0-6, so Wednesday is 3)
    my_time.tm_yday  =  time_list[7];  // Day of the year (0-365)
    my_time.tm_isdst =  time_list[8];   //daylight saving flag
    return my_time;
}
void convert_tm_to_timelist(int* time_list, struct tm my_time) {

    time_list[0] =my_time.tm_sec   ;     // Seconds
    time_list[1] =my_time.tm_min   ;    // Minutes
    time_list[2] =my_time.tm_hour  ;   // Hour (24-hour format)
    time_list[3] =my_time.tm_mday  ;   // Day of the month
    time_list[4] =my_time.tm_mon   ;     // Month (0-11, so October is 9)
    time_list[5] =my_time.tm_year  ;  // Years since 1900, so 2021 - 1900 = 121
    time_list[6] =my_time.tm_wday  ;    // Day of the week (0-6, so Wednesday is 3)
    time_list[7] =my_time.tm_yday  ;  // Day of the year (0-365)
    time_list[8] =my_time.tm_isdst ;   //daylight saving flag
}


int manhat_dist(int x1, int y1, int x2, int y2) {
    return abs(x1-x2) + abs(y1-y2);
}

int check_bounds(int row, int col) {
    return (row >= 0 && row <= nrows - 1) && (col >= 0 && col <= ncols - 1);
}

int check_bounds_coord(int* coord) {
    int row = coord[0];
    int col = coord[1];
    return (row >= 0 && row <= nrows - 1) && (col >= 0 && col <= ncols - 1);
}

int top_exists(int rank_node) {
    int coord[2];
    rank_to_coord(coord, rank_node, ncols, 0);
    int row = coord[0]; int col = coord[1];
    return check_bounds(row -1, col);
}

int bottom_exists(int rank_node) {
    int coord[2];
    rank_to_coord(coord, rank_node, ncols, 0);
    int row = coord[0]; int col = coord[1];
    return check_bounds(row +1, col);
}

int left_exists(int rank_node) {
    int coord[2];
    rank_to_coord(coord, rank_node, ncols, 0);
    int row = coord[0]; int col = coord[1];
    return check_bounds(row, col -1);
}

int right_exists(int rank_node) {
    int coord[2];
    rank_to_coord(coord, rank_node, ncols, 0);
    int row = coord[0]; int col = coord[1];
    return check_bounds(row, col + 1);
}
 
void log_report_to_file(FILE* pFile, int rank, int iter, struct nodeStruct value, int nearby_count, int* nearby_nodes_list, int** avail_2d_list, pthread_mutex_t* mutex_nearby, pthread_mutex_t* mutex_grid_avail) {
    char time_logged_buffer[80];
    char time_reported_buffer[80];

    // Get the current time as a time_t object
    time_t currentTime;
    time(&currentTime);
   
    struct tm localTime = *localtime(&currentTime);
    strftime(time_logged_buffer, sizeof(time_logged_buffer), "%a %Y-%m-%d %H:%M:%S", &localTime);
    struct tm reported_time = convert_timelist_to_tm(value.time_stamp); 
    strftime(time_reported_buffer, sizeof(time_reported_buffer), "%a %Y-%m-%d %H:%M:%S", &reported_time);

    int x_coord, y_coord;
    x_coord = rank / ncols;
    y_coord = rank % ncols;
    int port_avail = value.avail;
    int top[2] = {x_coord - 1, y_coord};
    int bot[2] = {x_coord + 1, y_coord};
    int left[2] = {x_coord, y_coord - 1};
    int right[2] = {x_coord, y_coord + 1};

    fprintf(pFile, "------------------------------------------------------------------------------------------------------------\n");
    fprintf(pFile, "ITERATION:  %d \n", iter);
    fprintf(pFile, "Logged time:            %-19s\n", time_logged_buffer);
    fprintf(pFile, "Alert reported time:    %-19s\n", time_reported_buffer);

    pthread_mutex_lock(mutex_grid_avail);
    printf("[MUTEX [grid_avail] - LOCKED]\n");
    print_visual_log(pFile, avail_2d_list);
    pthread_mutex_unlock(mutex_grid_avail);
    printf("[MUTEX [grid_avail] - UNLOCKED]\n");

    fprintf(pFile, "Reporting Node  Coord  Port Value  Available Port\n");
    fprintf(pFile, "%-15d  (%2d,%2d)     %-10d  %-15d\n", rank, x_coord, y_coord, k, port_avail);
    fprintf(pFile, "Adjacent Nodes  Coord  Port Value  Available Port\n");

    if (check_bounds_coord(top)) {
        fprintf(pFile, "%-15d  (%2d,%2d)     %-10d  %-15d\n", coord_to_rank(top, 0), top[0], top[1], value.neighbour_avail[0], k - value.neighbour_avail[0]);
    }
    if (check_bounds_coord(bot)) {
        fprintf(pFile, "%-15d  (%2d,%2d)     %-10d  %-15d\n", coord_to_rank(bot, 0), bot[0], bot[1], value.neighbour_avail[1], k - value.neighbour_avail[1]);
    }
    if (check_bounds_coord(left)) {
        fprintf(pFile, "%-15d  (%2d,%2d)     %-10d  %-15d\n", coord_to_rank(left, 0), left[0], left[1], value.neighbour_avail[2], k - value.neighbour_avail[2]);
    }
    if (check_bounds_coord(right)) {
        fprintf(pFile, "%-15d  (%2d,%2d)     %-10d  %-15d\n", coord_to_rank(right, 0), right[0], right[1], value.neighbour_avail[3], k - value.neighbour_avail[3]);
    }
    fprintf(pFile, "Nearby Nodes  Coord\n");

    pthread_mutex_lock(mutex_nearby);
    printf("[MUTEX [nearby] - LOCKED]\n");
    for (int i = 0; i < nearby_count; i++) {
        if (nearby_nodes_list[i] != -1) {
            x_coord = nearby_nodes_list[i] / ncols;
            y_coord = nearby_nodes_list[i] % ncols;
            fprintf(pFile, "%-15d  (%2d,%2d)\n", nearby_nodes_list[i], x_coord, y_coord);
        }
    }
    if (nearby_count == 0) {
        fprintf(pFile, "No nearby available nodes\n");
    }
    pthread_mutex_unlock(mutex_nearby);
    printf("[MUTEX [neraby] - UNLOCKED]\n");

    fprintf(pFile, "Communication Time (seconds): %lf\n",value.comm_time);
    fprintf(pFile, "------------------------------------------------------------------------------------------------------------\n\n");
}

void print_visual_log(FILE* pFile, int** avail_2d_list) {
    // Define ASCII characters for solid and empty blocks
    char solid_block[] = "■";
    char empty_block[] = "□";
    // Loop through each row
    for (int i = 0; i < nrows; i++) {
        // Loop through each column
        for (int j = 0; j < ncols; j++) {
            int value = avail_2d_list[i][j];
            char* block = (value < NODE_AVAIL_THRESHOLD) ? solid_block : empty_block;

            fprintf(pFile, "%s", block);

            // Print a space after each block except for the last column
            if (j < ncols - 1) {
                fprintf(pFile, " ");
            }
        }
        fprintf(pFile, "\n");
    }
}

double calc_time_taken(struct timespec start_time) {
    double time_taken;
    struct timespec curr_time;
    clock_gettime(CLOCK_MONOTONIC, &curr_time); 
    time_taken = (curr_time.tv_sec - start_time.tv_sec) * 1e9; 
    time_taken = (time_taken + (curr_time.tv_nsec - start_time.tv_nsec)) * 1e-9; 
    return time_taken;
}

void get_time_for_nodeStruct(int* time_stamp) {
    time_t currentTime;
    time(&currentTime);
    struct tm *localTime = localtime(&currentTime);
    convert_tm_to_timelist(time_stamp, *localTime);

}
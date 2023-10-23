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
#define PERIOD_CHECK 5                     //every 10 seconds base checks
#define TIME_MSG_VALID 30
#define TIME_CHECK_THRESHOLD 0.001
#define MAX_ITER 10
#define NODE_AVAIL_PERCENTAGE 0.8
#define TIME_LOG_VALID 20

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
#define DEBUG 0
#define CHECK_STUFF 1

#define CART_SHIFT 1 
//////////////////////////////////////////////////////////////////////////////////////
// GLOBAL VARIABLES 
//////////////////////////////////////////////////////////////////////////////////////
int nrows, ncols, k;
int rank_base;
int size;
int node_avail_threshold;
int non_imm_neigh_list_len;
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
struct tm get_time_tm(void);
int node_avail(int val);
int node_unavail(int val);
int node_no_comm(int val);


int main(int argc, char *argv[]) {

    //set the time to Australian/Sydney time 
    setenv("TZ", "Australia/Sydney", 1);
    tzset();

    struct nodeStruct values;
    int size, my_rank;
    //start up initial MPI environment 
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    
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
    rank_base = size -1; 
    non_imm_neigh_list_len = 2*(nrows + ncols);
    node_avail_threshold = (int)(NODE_AVAIL_PERCENTAGE * k);


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
    int temp_log_flag;

    if(pid == 0) { // the main communications thread of the base is allowed to read the ptr_iter because only it can modify it anyway 
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
                rank_to_coord(coord, n, ncols, 0);
                // probe for any messages from this node
                MPI_Iprobe(n, TAG_MSG_NODE_TO_BASE, MPI_COMM_WORLD, &flag_list[n], &status_list[n]);
                if(flag_list[n]) {
                    
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
            MPI_Send(&termination_msg, 1, MPI_INT, i, TAG_TERMINATE, comm_world);
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

            pthread_mutex_lock(ptr_mutex_iter);
            if(!BASE_OFF) {printf("[MUTEX [iter] - LOCKED]\n");}
            temp_iter = *ptr_iter;
            pthread_mutex_unlock(ptr_mutex_iter);
            if(!BASE_OFF) {printf("[MUTEX [iter] - UNLOCKED]\n");}

            for(int n = 0; n > size - 1; n++) {
                pthread_mutex_lock(ptr_mutex_log);
                temp_log_flag = log_1d_flag_list[n];
                pthread_mutex_unlock(ptr_mutex_log);

                //check if there is a flag 
                if(temp_log_flag) {
                    //mutex - using shared data between the threads at base 
                    pthread_mutex_lock(ptr_mutex_log);
                    if(!BASE_OFF) {printf("[MUTEX [log] - LOCKED]\n");}
                    log_1d_flag_list[n] = 0; //unset flag
                    value = log_1d_struct_list[n];  
                    pthread_mutex_unlock(ptr_mutex_log);
                    if(!BASE_OFF) {printf("[MUTEX [log] - UNLOCKED]\n");}

                    log_report_to_file(p_log, n, temp_iter, value, nearby_1d_flag_list[n], nearby_2d_list[n], avail_2d_list,  ptr_mutex_nearby, ptr_mutex_grid_avail); 
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
    //initialise all nodes to have avail = k 
    for(int i = 0 ; i < nrows; i++) {
        for(int j = 0; j < ncols; j++) {
            avail_2d_list[i][j] =  k;
        }
    }
    //1d array to store flags for all nodes 
    int* log_1d_flag_list = (int*)calloc(size-1, sizeof(int));

    //1d array to store received log from a node 
    struct nodeStruct* log_1d_struct_list = (struct nodeStruct*)calloc(size-1, sizeof(struct nodeStruct));

    //1d array to flag if there is a list for nearby neighbours - essentially contains th count 
    int* nearby_1d_flag_list = (int*)calloc(size-1, sizeof(int));

    //2d array to store nearby nodes for each node 
    int nearby_nodes_list_len = 2*(nrows + ncols);
    int** nearby_2d_list = (int**)calloc(size-1, sizeof(int*));
    for(int i = 0; i < size - 1; i++) {
        nearby_2d_list[i] = (int*)calloc(nearby_nodes_list_len, sizeof(int));
    }

    int terminate_all = 0;
    int run_iter = 1;
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
    if(!BASE_OFF) {printf("---------port avail threshold = %d----------------\n", node_avail_threshold);}

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

    //shared table for each port (each node has k ports)
    int size_shared_list = k;
    struct tm* shared_table_timestamp_1d_list = (struct tm*)calloc(size_shared_list, sizeof(struct tm)); 
    int* shared_table_1d_list = (int*)calloc(size_shared_list, sizeof(int)); 

    
    MPI_Comm_rank(comm_world, &rank_world);
    MPI_Comm_rank(comm_group, &rank_comm);
    MPI_Comm_rank(comm2d, &rank_grid);
    MPI_Cart_coords(comm2d, rank_grid, nDims, coord);
    char fname_node[80];
    snprintf(fname_node, sizeof(fname_node), "node_logs/node_rank_%d_coord_%d-%d.txt", rank_world, coord[0], coord[1]);
    FILE* pFile;
    pFile = fopen(fname_node, "w");
    if (pFile == NULL) {
        perror("Error opening file");
        return 1;
    }

    int port_avail_list_counter = 0;

    int termination = 0;
    int run_iter = 1;
    int iter;
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

    time_t curr_time, temp_time;
    struct tm start_time_tm[4], time_tm;
    double time_taken = 0.0;
    int time_stamp[TIMESTAMP_NO_ITEMS];

    MPI_Request send_request[4];
    MPI_Request receive_request[4];
    MPI_Status temp_status;

    int disp = 1;
    int rank_world, rank_group, rank_grid;
    int size;
    MPI_Comm_size(comm_world, &size);
    MPI_Comm_rank(comm2d, &rank_grid);
    MPI_Comm_rank(comm_world, &rank_world);
    MPI_Comm_rank(comm_group, &rank_group);
    int rank_base = size-1;

    int nbr_i_lo, nbr_i_hi; //for top and bottom  
    int nbr_j_lo, nbr_j_hi; //for left and right
    int neigh_rank_list[4] = {nbr_i_lo, nbr_i_hi, nbr_j_lo, nbr_j_hi};
    MPI_Cart_shift(comm2d, SHIFT_ROW, CART_SHIFT, &nbr_i_lo, &nbr_i_hi);  //lo = top, hi = bot
    MPI_Cart_shift(comm2d, SHIFT_COL, CART_SHIFT, &nbr_j_lo, &nbr_j_hi); //lo = left, hi = right
    int coord[2];
    MPI_Cart_coords(comm2d, rank_grid, NO_DIMS, &coord);
    int neigh_exist_list[4], neigh_avail_list[4];
    struct nodeStruct nodeStruct_to_base;
    int row = coord[0], col = coord[1];

    int top_exists = check_bounds(row-1, col); neigh_exist_list[0] = top_exists; nodeStruct_to_base.neighbour_exists[0] = top_exists; neigh_avail_list[0] = top_exists;
    int bottom_exists = check_bounds(row+1, col); neigh_exist_list[1] = bottom_exists; nodeStruct_to_base.neighbour_exists[1] = bottom_exists; neigh_avail_list[1] = bottom_exists;
    int left_exists = check_bounds(row, col-1); neigh_exist_list[2] = left_exists; nodeStruct_to_base.neighbour_exists[2] = left_exists; neigh_avail_list[2] = left_exists;
    int right_exists = check_bounds(row, col+1); neigh_exist_list[3] = right_exists; nodeStruct_to_base.neighbour_exists[3] = right_exists; neigh_avail_list[3] = right_exists;

    int temp_termination, temp_iter, flag_terminate, flag_non_imm_count, flag_non_imm_list, port_used_old, port_used_new, temp_run_iter;
    int non_imm_count = 0, any_neigh_avail = 0;
    int exp_reply_from_any_neigh = 0, expecting_reply_base = 0, flag_recv_non_imm_neigh_count = 0, flag_recv_non_imm_list = 0;
    int flag_plea_neigh[4], flag_reply_neigh[4], recv_plea_neigh[4], recv_reply_neigh[4], neigh_ports_free_list[4];
    port_used_old = 0;
    int exp_reply_neigh[4]; for(int i = 0;i < 4; i++) {exp_reply_neigh[i] = 0;}
    int temp_avail;
    int non_imm_neigh_list[2*(nrows+ncols)]; 
    struct tm time_tm_epoch;
    struct tm start_time_send[4]; 
    int service, temp_shared_table_counter;
    
    struct timespec start_time;
    if(pid == k) { //the node itself
        pthread_mutex_lock(ptr_mutex_iter);
        *ptr_iter = 1;
        pthread_mutex_unlock(ptr_mutex_iter);

        pthread_mutex_lock(ptr_mutex_run_iter);
        *ptr_run_iter = 0;
        pthread_mutex_unlock(ptr_mutex_run_iter);
        
        pthread_mutex_lock(ptr_mutex_termination);
        *ptr_termination = 0;
        pthread_mutex_unlock(ptr_mutex_termination);

        clock_gettime(CLOCK_MONOTONIC, &start_time); 
        while(1) { 
            //calculate time_taken
            time_taken = calc_time_taken(start_time);

            //check for termination message from base
                //probe   
            MPI_Iprobe(rank_base, TAG_TERMINATE, comm_world, &flag_terminate, &temp_status);
                //if probed, recv  
            if(flag_terminate) {MPI_Recv(&temp_termination, 1, MPI_INT, rank_base, TAG_TERMINATE, comm_world, &temp_status); pthread_mutex_lock(ptr_mutex_termination); *ptr_termination = 1; pthread_mutex_unlock(ptr_mutex_termination); break;}
            
            //check if we going to run in this iteration
                // if time_taken % PERIODIC_CHECK ....
            if(fmod(time_taken, (double) PERIOD_CHECK) >= TIME_CHECK_THRESHOLD) {pthread_mutex_lock(ptr_mutex_run_iter); *ptr_run_iter = 0; pthread_mutex_unlock(ptr_mutex_run_iter); continue;}  
            else {pthread_mutex_lock(ptr_mutex_run_iter); *ptr_run_iter = 1;pthread_mutex_unlock(ptr_mutex_run_iter);} //run this iteration 

            pthread_mutex_lock(ptr_mutex_iter);
            temp_iter = *ptr_iter;
            pthread_mutex_unlock(ptr_mutex_iter);

            
            pthread_mutex_lock(ptr_mutex_avail);
            temp_avail = *ptr_avail;
            pthread_mutex_unlock(ptr_mutex_avail);

            if(!NODES_OFF) {printf("--------------------------------------------------------[ITER] node %d (%d,%d): %d------------------ avail = %d-------------------------\n\n", rank_grid, coord[0], coord[1], temp_iter, temp_avail);}
            // if(rank_grid == 3) {printf("--------------------------------------------------------iter == %d------------------------------------------\n\n", temp_iter);}

            //check for any messages from base: replies about non-immediate neighbours 
            MPI_Iprobe(rank_base, TAG_BASE_NODE_NON_IMM_COUNT, comm_world, &flag_non_imm_count, &temp_status);
            MPI_Iprobe(rank_base, TAG_NON_IMM_NEIGHBOURS, comm_world, &flag_non_imm_list, &temp_status);
            if(flag_non_imm_count) {MPI_Recv(&non_imm_count, 1, MPI_INT, rank_base, TAG_BASE_NODE_NON_IMM_COUNT, comm_world, &temp_status); flag_recv_non_imm_neigh_count = 1;} 
            if(flag_non_imm_list) {MPI_Recv(&non_imm_neigh_list, non_imm_neigh_list_len, MPI_INT, rank_base, TAG_NON_IMM_NEIGHBOURS, comm_world, &temp_status); flag_recv_non_imm_list = 1;}

            for(int i = 0; i < 4; i++) {exp_reply_from_any_neigh = exp_reply_from_any_neigh || exp_reply_neigh[i]; neigh_avail_list[i] = 0;}
            any_neigh_avail = 0;
            
            if(rank_grid == 0 && DEBUG) {printf("--------------------------------------------------------rank = %d STUCK 0-------------------------------------------\n\n", rank_grid);}

            //check for pleas from neighbouring nodes, and respond to the plea 
            if(top_exists)                            { MPI_Iprobe(nbr_i_lo, TAG_PLEA, comm2d, &flag_plea_neigh[0], &temp_status);}  // if top exists: check
            if(bottom_exists)                         { MPI_Iprobe(nbr_i_hi, TAG_PLEA, comm2d, &flag_plea_neigh[1], &temp_status);}  // if bottom exists: check   
            if(left_exists)                           { MPI_Iprobe(nbr_j_lo, TAG_PLEA, comm2d, &flag_plea_neigh[2], &temp_status);}  // if left exists: check   
            if(right_exists)                          { MPI_Iprobe(nbr_j_hi, TAG_PLEA, comm2d, &flag_plea_neigh[3], &temp_status);}  // if right exists: check   
            if(flag_plea_neigh[0])                   { MPI_Recv(&recv_plea_neigh[0], 1, MPI_INT, nbr_i_lo, TAG_PLEA, comm2d, &temp_status);} 
            if(flag_plea_neigh[1])                   { MPI_Recv(&recv_plea_neigh[1], 1, MPI_INT, nbr_i_hi, TAG_PLEA, comm2d, &temp_status);}
            if(flag_plea_neigh[2])                   { MPI_Recv(&recv_plea_neigh[2], 1, MPI_INT, nbr_j_lo, TAG_PLEA, comm2d, &temp_status);}
            if(flag_plea_neigh[3])                   { MPI_Recv(&recv_plea_neigh[3], 1, MPI_INT, nbr_j_hi, TAG_PLEA, comm2d, &temp_status);}
            //check for any plea replies if we are expecting any 
            if(top_exists   )                         { MPI_Iprobe(nbr_i_lo, TAG_PLEA_REPLY, comm2d, &flag_reply_neigh[0], &temp_status);}  // if top exists: check
            if(bottom_exists)                         { MPI_Iprobe(nbr_i_hi, TAG_PLEA_REPLY, comm2d, &flag_reply_neigh[1], &temp_status);}  // if bottom exists: check   
            if(left_exists  )                         { MPI_Iprobe(nbr_j_lo, TAG_PLEA_REPLY, comm2d, &flag_reply_neigh[2], &temp_status);}  // if left exists: check   
            if(right_exists )                         { MPI_Iprobe(nbr_j_hi, TAG_PLEA_REPLY, comm2d, &flag_reply_neigh[3], &temp_status);}  // if right exists: check   
            if(flag_reply_neigh[0])                  { MPI_Recv(&recv_reply_neigh[0], 1, MPI_INT, nbr_i_lo, TAG_PLEA_REPLY, comm2d, &temp_status);} 
            if(flag_reply_neigh[1])                  { MPI_Recv(&recv_reply_neigh[1], 1, MPI_INT, nbr_i_hi, TAG_PLEA_REPLY, comm2d, &temp_status);}
            if(flag_reply_neigh[2])                  { MPI_Recv(&recv_reply_neigh[2], 1, MPI_INT, nbr_j_lo, TAG_PLEA_REPLY, comm2d, &temp_status);}
            if(flag_reply_neigh[3])                  { MPI_Recv(&recv_reply_neigh[3], 1, MPI_INT, nbr_j_hi, TAG_PLEA_REPLY, comm2d, &temp_status);}

            if(rank_grid == 0 && DEBUG) {printf("--------------------------------------------------------rank = %d STUCK 1-------------------------------------------\n\n", rank_grid);}

            if(temp_avail < node_avail_threshold) { //if at capacity 
                if(rank_grid == 0 && DEBUG) {printf("--------------------------------------------------------rank = %d STUCK 2-------------------------------------------\n\n", rank_grid);}

                if(expecting_reply_base) { //if you've sent a plea out to the base, you need to wait until it's received 
                    if(rank_grid == 0 && DEBUG) {printf("--------------------------------------------------------rank = %d STUCK 3-------------------------------------------\n\n", rank_grid);}

                    if(flag_recv_non_imm_neigh_count && flag_recv_non_imm_list) { //received both the count + non-imm neighbours from base 
                        if(rank_grid == 0 && DEBUG) {printf("--------------------------------------------------------rank = %d STUCK 4-------------------------------------------\n\n", rank_grid);}
                        //log to a file - not done here (performed by the other thread)
                        //reset 
                        for(int i = 0; i < non_imm_count; i++) {non_imm_neigh_list[i] = -1;}
                        flag_recv_non_imm_neigh_count = 0;
                        non_imm_count = 0;
                        flag_recv_non_imm_neigh_count = 0;
                        expecting_reply_base = 0;
                    }
                    else if(flag_recv_non_imm_neigh_count && non_imm_count == 0) {
                        if(rank_grid == 0 && DEBUG) {printf("--------------------------------------------------------rank = %d STUCK 5-------------------------------------------\n\n", rank_grid);}
                        //log to node file --not performed here
                        flag_recv_non_imm_neigh_count = 0;
                    }
                }
                else  {  //if not expecting any replies from the base
                    for(int i = 0; i < 4; i++) { //go through each of the nodes neighbours 
                        if(rank_grid == 0 && DEBUG) {printf("--------------------------------------------------------rank = %d STUCK 6-------------------------------------------\n\n", rank_grid);}
                        if(recv_plea_neigh[i] != -1) { //received a plea 
                            neigh_avail_list[i] = 0; //not available 
                            neigh_ports_free_list[i] = recv_plea_neigh[i];
                        }
                        else if(exp_reply_neigh[i] && recv_reply_neigh[i] != -1) {  //check if received plea reply (if expecting it)
                            if(rank_grid == 0 && DEBUG) {printf("--------------------------------------------------------rank = %d STUCK 7-------------------------------------------\n\n", rank_grid);}
                            neigh_avail_list[i] = node_avail(recv_reply_neigh[i]); //check if avail;
                            neigh_ports_free_list[i] = recv_reply_neigh[i];
                            exp_reply_neigh[i] = 0; //not expecting anymore 
                            time(&curr_time);
                            temp_time = mktime(&start_time_tm[i]); //convert from struct tm to time_t 
                            nodeStruct_to_base.comm_time[i] = difftime(curr_time, temp_time); 
                        }
                        else if(!exp_reply_neigh[i] && neigh_exist_list[i]) {
                            if(rank_grid == 0 && DEBUG) {printf("--------------------------------------------------------rank = %d STUCK 8-------------------------------------------\n\n", rank_grid);}
                            MPI_Isend(&temp_avail, 1, MPI_INT, neigh_rank_list[i], TAG_PLEA, comm2d, &send_request[i]);  //send a plea out to a neighbour 
                            start_time_tm[i] = get_time_tm(); //get track of when you started so you can see the communication time 
                            exp_reply_neigh[i] = 1; 
                        }
                    }
                    any_neigh_avail = 0;
                    for(int i= 0; i < 4; i++) {any_neigh_avail = any_neigh_avail || neigh_avail_list[i];}
                    exp_reply_from_any_neigh = 0;
                    for(int i = 0; i < 4; i++) {exp_reply_from_any_neigh = exp_reply_from_any_neigh || exp_reply_neigh[i];}

                    if(rank_grid == 0 && DEBUG) {printf("--------------------------------------------------------rank = %d STUCK 9-------------------------------------------\n\n", rank_grid);}
                    if(!exp_reply_from_any_neigh) { //send to base as a plea
                        nodeStruct_to_base.avail = temp_avail;
                        for(int i = 0; i < 4; i++) {nodeStruct_to_base.neighbour_avail[i] = neigh_ports_free_list[i]; recv_reply_neigh[i] = -1; recv_plea_neigh[i] = -1;}
                        time_tm = get_time_tm();
                        convert_tm_to_timelist(&time_stamp, time_tm);
                        if(rank_grid == 0 && CHECK_STUFF) {printf("--------------------------------------------------------rank = %d STUCK 10-------------------------------------------\n\n", rank_grid);}
                        for(int i = 0; i < TIMESTAMP_NO_ITEMS; i++) {nodeStruct_to_base.time_stamp[i] =  time_stamp[i];}

                        if(!any_neigh_avail) {
                            nodeStruct_to_base.tag = TAG_PLEA;
                            MPI_Isend(&nodeStruct_to_base, 1, Valuetype, rank_base, TAG_PLEA, comm_world, send_request); 
                        }
                        else{
                            nodeStruct_to_base.tag = TAG_LOG;
                            MPI_Isend(&nodeStruct_to_base, 1, Valuetype, rank_base, TAG_LOG, comm_world, send_request); 
                        }
                        
                    }
                }
            }
            else { 
                if(rank_grid == 0 && DEBUG) {printf("--------------------------------------------------------rank = %d STUCK 11-------------------------------------------\n\n", rank_grid);}
                for(int i = 0; i < 4; i++) {exp_reply_neigh[i] = 0; recv_reply_neigh[i] = -1; recv_plea_neigh[i] = -1; nodeStruct_to_base.comm_time[i] = 0.0; neigh_ports_free_list[i] = k;}
                flag_recv_non_imm_neigh_count = 0;
                non_imm_count = 0;
                flag_recv_non_imm_neigh_count = 0;
                expecting_reply_base = 0;
                exp_reply_from_any_neigh = 0;
                flag_recv_non_imm_list = 0 ;
            }
            //increment ite
            pthread_mutex_lock(ptr_mutex_iter);
            *ptr_iter = *ptr_iter + 1;
            pthread_mutex_unlock(ptr_mutex_iter);   
        }
    }
    else {  // all the charging ports
        while(1) {
            //check for termination message from the pid = k th node (main node)
            pthread_mutex_lock(ptr_mutex_termination);
            temp_termination = *ptr_termination;
            pthread_mutex_unlock(ptr_mutex_termination);
            if(temp_termination) {
                break;
            }

            //check if we run in this iteration
            pthread_mutex_lock(ptr_mutex_run_iter);
            temp_run_iter = *ptr_run_iter;
            pthread_mutex_unlock(ptr_mutex_run_iter);
            if(!temp_run_iter){
                continue;
            }
            pthread_mutex_lock(ptr_mutex_iter);
            temp_iter = *ptr_iter;
            pthread_mutex_unlock(ptr_mutex_iter);

            pthread_mutex_lock(ptr_mutex_avail);
            temp_avail = *ptr_avail;
            pthread_mutex_unlock(ptr_mutex_avail);
            
            port_used_new = (int) (((rand() % (UPPER_VAL_PROB - LOWER_VAL_PROB + 1)) + LOWER_VAL_PROB) < PROB_PORT_SWITCHES);

            service = 1;
            if(port_used_old && !port_used_new) {
                pthread_mutex_lock(ptr_mutex_avail);
                *ptr_avail = *ptr_avail + 1;
                temp_avail = *ptr_avail;
                pthread_mutex_unlock(ptr_mutex_avail);
            }
            else if(!port_used_old && port_used_new)  {
                pthread_mutex_lock(ptr_mutex_avail);
                *ptr_avail = *ptr_avail - 1;
                temp_avail = *ptr_avail;
                pthread_mutex_unlock(ptr_mutex_avail);
            }
            else {
                service = 0;
            }
            if(service == 1) {
                time_tm = get_time_tm();
                pthread_mutex_lock(ptr_mutex_shared_list);
                temp_shared_table_counter = *ptr_port_avail_list_counter;
            
                if(temp_shared_table_counter > k-1) {
                    for(int j = 1; j < k; j ++) {
                        shared_table_1d_list[j-1] = shared_table_1d_list[j];
                        shared_table_timestamp_1d_list[j-1] = shared_table_timestamp_1d_list[j];
                    }
                    temp_shared_table_counter--;
                }
                shared_table_1d_list[temp_shared_table_counter] = temp_avail;
                shared_table_timestamp_1d_list[temp_shared_table_counter] = time_tm;
                temp_shared_table_counter++;
                *ptr_port_avail_list_counter = temp_shared_table_counter;
                pthread_mutex_unlock(ptr_mutex_shared_list);

                log_node_file_shared_list(pFile, ptr_mutex_shared_list, shared_table_timestamp_1d_list, shared_table_1d_list, ptr_port_avail_list_counter);
            }
            // if(!NODES_OFF) {printf(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>---------------[ITER] node %d (%d,%d): %d STUCK IN port %d -------------------------------------------\n\n", rank_grid, coord[0], coord[1], temp_iter, pid);}
            
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
                else if(time_diff < TIME_LOG_VALID && avail_2d_list[row][col] >= node_avail_threshold) {
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

                if(time_diff < TIME_LOG_VALID && avail_2d_list[row][col] >= node_avail_threshold) {
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

                if(time_diff < TIME_LOG_VALID && avail_2d_list[row][col] >= node_avail_threshold) {
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
                if(time_diff < TIME_LOG_VALID && avail_2d_list[row][col] >= node_avail_threshold) {
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
int node_avail(int val) {
    return val >= 0 && val >= node_avail_threshold;
}
int node_unavail(int val) {
    return val >= 0 && val < node_avail_threshold;
}
int node_no_comm(int val) {
    return val == -1;
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
    const char* tag = (value.tag == TAG_PLEA) ? "PLEA" : "REPORT";

    fprintf(pFile, "------------------------------------------------------------------------------------------------------------\n");
    fprintf(pFile, "ITERATION:  %d \n", iter);
    fprintf(pFile, "Tag:   %s\n", tag);
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
    
    if(value.tag == TAG_PLEA) { // only print nearby nodes if it was a TAG_PLEA which the base would have responded to already 
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
    }
    

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
            char* block = (value < node_avail_threshold) ? solid_block : empty_block;

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

struct tm get_time_tm(void) {
    time_t curr_time;
    time(&curr_time);
    struct tm time_tm;
    time_tm = *localtime(&curr_time);
    return time_tm;
}
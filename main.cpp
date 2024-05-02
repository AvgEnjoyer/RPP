#include <vector>
#include <iostream>
#include <mpi.h>
#include <chrono>

const int CONSTANTA = 10'000'000;


void print_arr(std::vector<int>& nums) {
    for (int num : nums) {
        std::cout << num << ", ";
    }
    std::cout << std::endl;
}

void merge(std::vector<int>& nums, std::vector<int>& left, std::vector<int>& right) {
    int i = 0, j = 0, k = 0;
    while (i < left.size() && j < right.size()) {
        if (left[i] < right[j]) {
            nums[k++] = left[i++];
        }
        else {
            nums[k++] = right[j++];
        }
    }
    while (i < left.size()) {
        nums[k++] = left[i++];
    }
    while (j < right.size()) {
        nums[k++] = right[j++];
    }
}

void mergeSort(std::vector<int>& nums) {
    if (nums.size() > 1) {
        int mid = nums.size() / 2;
        std::vector<int> left(nums.begin(), nums.begin() + mid);
        std::vector<int> right(nums.begin() + mid, nums.end());
        mergeSort(left);
        mergeSort(right);
        merge(nums, left, right);
    }
}


void mergeSortMPI(std::vector<int>& nums) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size == 1) {
        mergeSort(nums);
        return;
    }

    int local_size = nums.size() / size;
    int remaining = nums.size() % size;

    std::vector<int> local_nums(local_size + (rank < remaining ? 1 : 0));
    MPI_Scatter(nums.data(), local_size + (rank < remaining ? 1 : 0), MPI_INT,
        local_nums.data(), local_size + (rank < remaining ? 1 : 0), MPI_INT, 0, MPI_COMM_WORLD);

    mergeSort(local_nums);

    std::vector<int> merged;
    if (rank == 0) {
        merged.resize(nums.size());
    }

    // Gather the sorted segments on the root process
    std::vector<int> gathered_nums(nums.size());
    MPI_Gather(local_nums.data(), local_size + (rank < remaining ? 1 : 0), MPI_INT,
        gathered_nums.data(), local_size + (rank < remaining ? 1 : 0), MPI_INT, 0, MPI_COMM_WORLD);

    // Root process concatenates and sorts the gathered segments
    if (rank == 0) {
        // The start and end indices of each segment
        std::vector<int> starts(size, 0), ends(size, 0);
        for (int i = 0; i < size; ++i) {
            starts[i] = i * local_size + std::min(i, remaining);
            ends[i] = starts[i] + local_size + (i < remaining ? 1 : 0);
        }

        // Merge the sorted segments
        for (int i = 0; i < nums.size(); ++i) {
            int min_val = INT_MAX;
            int min_index = -1;

            // Find the smallest element among the first elements of the segments
            for (int j = 0; j < size; ++j) {
                if (starts[j] < ends[j] && gathered_nums[starts[j]] < min_val) {
                    min_val = gathered_nums[starts[j]];
                    min_index = j;
                }
            }

            // Add the smallest element to the merged array and move the start index of the corresponding segment
            merged[i] = min_val;
            starts[min_index]++;
        }

        nums = merged;
    }
}


void exec_mpi(int argc, char** argv) {
    int world_rank;
    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    std::vector<int> nums(CONSTANTA);
    std::vector<int> nums_copy(CONSTANTA);
    int max = -1;
    if (world_rank == 0) {
        std::cout << "Array initialization... " << std::endl;
        for (int i = 0; i < CONSTANTA; ++i) {
            int rand_num = std::rand();
            nums[i] = rand_num;
            nums_copy[i] = rand_num;
            if (rand_num > max)
                max = rand_num;
        }
        std::cout << "Done... " << std::endl;
    }

    MPI_Bcast(nums.data(), nums.size(), MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(nums_copy.data(), nums_copy.size(), MPI_INT, 0, MPI_COMM_WORLD);

    auto start1 = std::chrono::high_resolution_clock::now();
    mergeSortMPI(nums);
    auto stop1 = std::chrono::high_resolution_clock::now();
    auto duration1 = std::chrono::duration_cast<std::chrono::microseconds>(stop1 - start1);

    if (world_rank == 0) {
        std::cout << "Time taken by MPI: " << duration1.count() << " microseconds" << std::endl;
        std::cout << "First 3 items of nums: " << nums[0] << ", " << nums[1] << ", " << nums[2] << std::endl;
        std::cout << "Last 3 items of nums: " << nums[CONSTANTA - 3] << ", " << nums[CONSTANTA - 2] << ", " << nums[CONSTANTA - 1] << std::endl;
        std::cout << "size: " << nums.size();
    }

    MPI_Finalize();
}


int main(int argc, char** argv) {
    exec_mpi(argc,argv);
    return 0;
}



//int main(int argc, char** argv) {
//    exec_mpi(argc, argv);
//    return 0;
//}
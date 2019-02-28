void initialData(float* ip, int size){
  time_t t;
  srand((unsigned int)time(&t));
  for(int i = 0; i<size; i++){
    ip[i] = (float)(rand() & 0xFF)/10.0f;
  }
}

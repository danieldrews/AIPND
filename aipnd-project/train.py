import argparse

def get_input_args():
  ag = argparse.ArgumentParser(description="Program setup variables")
  ag.add_argument('--learning_rate', default=0.01)
  ag.add_argument('--hidden_units', default=512)
  ag.add_argument('--epochs ', default=1)

def main():
  print('main')
  in_arg = get_input_args()

if __name__ == "__main__":
  main()
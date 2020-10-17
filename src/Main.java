import java.io.*;
import java.util.*;


public class Main {
	public static void main(String arg[]) {
		List<List<List<String>>> list = new ArrayList<List<List<String>>>();
		int Vmax = 0;
		int Kmax = 0;
		int argOfV = 0;
		int agV[] = new int[arg.length];		//何変数の引数か知りたい
		for(int n = 0; n < arg.length;n++) {
			agV[n] = 0;
			String ag[] = arg[n].split(" ",0);
			for(int a = 0; a < ag.length; a++) {
				if(ag[a].startsWith("?")) {
					agV[n]++;
				}
			}
			if(agV[n] > 0) argOfV++;
		}
		if(arg.length > 1) {
			for(int n = 0; n < arg.length-1; n++) {
				if(agV[n] > agV[n+1]) {
					agV[n+1] = agV[n];
					String tmp = arg[n+1];
					arg[n+1] = arg[n];
					arg[n] = tmp;
				}
			}
		}
		
		try {    // ファイル読み込みに失敗した時の例外処理のためのtry-catch構文
            String fileName = "dataset_example.txt"; // ファイル名指定
            
            for(int n = 0; n< arg.length; n++) 
	        {
            	int countV = 0;
            	list.add(new ArrayList<List<String>>());
            // 文字コードUTF-8を指定してBufferedReaderオブジェクトを作る
            BufferedReader in = new BufferedReader(new InputStreamReader(new FileInputStream(fileName), "UTF-8"));
	            // 変数lineに1行ずつ読み込むfor文
	            for (String line = in.readLine(); line != null; line = in.readLine()) {
	            	Unifier u= new Unifier(arg[n],line);
	            	
			            	if(u.tf == true) 	
			            	{	
			            		//System.out.println(line);
			            		if(u.getV()>0) {
			            			HashMap<String,String> vars =u.getVars();
			            			ArrayList var = new ArrayList(vars.values());
			            			list.get(n).add(var);
			            			//System.out.println(var.toString());
			            			//list.get(n).get(countV).add(var);
			            			countV++;
			            			if(var.size()>Kmax) Kmax = var.size();
			            			if(countV>Vmax) Vmax = countV;
			            		}
			            			
			            	}
	            }
            in.close();  // ファイルを閉じる
	        }
        } catch (IOException e) {
            e.printStackTrace(); // 例外が発生した所までのスタックトレースを表示
        }
		if(argOfV<2) {
			//System.out.println("wa");
			int cnt = 0;
			String ans[][] = new String[argOfV*Vmax][Kmax];
			//System.out.println(list.get().size());
			for(int i = 0;i<list.get(list.size()-1).size();i++) {
				//System.out.println("i");
				for(int k = 0;k<list.get(list.size()-1).get(i).size();k++) {
					ans[cnt][k]=list.get(list.size()-1).get(i).get(k);
					//System.out.println("k");
				}
				cnt++;
			}
			for(int i = 0;i<ans.length;i++) {
				System.out.print("(");
				for(int k = 0;k<ans[0].length;k++) {
					if(k>0)	System.out.print(", ");
					System.out.print(ans[i][k]);					
				}
				System.out.println(")");
			}
		}else {
			String ans[][] = new String[list.size()*Vmax][Kmax];	//argOfVに差し替え確認
			int cnt = 0;
			for(int i = 0;i<list.size();i++) {
				for(int j = 0; j < list.get(i).size();j++) {
					for(int k = 0; k < list.get(i).get(j).size(); k++) {
						for(int n = list.size()-i-1;n > 0;n--) {
							for(int m = list.get(i+n).size()-j-1;m > 0;m--) {
								if(list.get(i).get(j).get(k).equals(list.get(i+n).get(j+m).get(k))) {
									for(int h = 0;h<list.get(i+n).get(j+m).size();h++) {
										ans[cnt][h] = list.get(i+n).get(j+m).get(h);
									}						
									cnt++;
								}
							}
						}
						
					}
				}
			}
		/*for(int i = 0;i<ans.length;i++) {
			for(int j = 0;j<ans[i].length;j++) {
				System.out.println(i+" "+j+ans[i][j]);
			}
		}*/
			
			for(int i = 0;i<ans.length;i++) {
				if(ans[i][0]==null) break;
				System.out.print("(");
				for(int j = 0;j<ans[i].length;j++) {
					if(ans[i][j]==null) break;
					if(j>0)	System.out.print(", ");
					System.out.print(ans[i][j]);			
				}
				System.out.println(")");
			}
		}	
	}
}

/***
  Matching Program written



  �ϐ�:�O�ɁH������D  

  Examle:
  % Matching "Takayuki" "Takayuki"
  true

  % Matching "Takayuki" "Takoyuki"
  false

  % Matching "?x am Takayuki" "I am Takayuki"
  ?x = I .

  % Matching "?x is ?x" "a is b"
  false

  % Matching "?x is ?x" "a is a"
  ?x = a .


  Mating �́C�p�^�[���\���ƒʏ�\���Ƃ��r���āC�ʏ�\����
  �p�^�[���\���̗�ł��邩�ǂ����𒲂ׂ�D
  Unify �́C���j�t�B�P�[�V�����ƍ��A���S���Y�����������C
  �p�^�[���\�����r���Ė����̂Ȃ�����ɂ���ē���Ɣ��f
  �ł��邩�ǂ����𒲂ׂ�D
  
  ***/

import java.lang.*;
import java.util.*;

/**
 * �}�b�`���O�̃v���O����
 * 
 */
class Matching {
    public static void main(String arg[]){
	if(arg.length != 2){
	    System.out.println("Usgae : % Matching [string1] [string2]");
	}
	System.out.println((new Matcher()).matching(arg[0],arg[1]));
    }
}

/**
 * ���ۂɃ}�b�`���O���s���N���X
 * 
 */
class Matcher {
    StringTokenizer st1;
    StringTokenizer st2;
    HashMap<String,String> vars;
    
    Matcher(){
	vars = new HashMap<String,String>();
    }

    public boolean matching(String string1,String string2,HashMap<String,String> bindings){
	this.vars = bindings;
	if(matching(string1,string2)){
	    return true;
	} else {
	    return false;
	}
    }
    
    public boolean matching(String string1,String string2){
	//System.out.println(string1);
	//System.out.println(string2);
	
	// �����Ȃ琬��
	if(string1.equals(string2)) return true;
	
	// �e�X�g�[�N���ɕ�����
	st1 = new StringTokenizer(string1);
	st2 = new StringTokenizer(string2);

	// �����قȂ����玸�s
	if(st1.countTokens() != st2.countTokens()) return false;
		
	// �萔���m
	for(int i = 0 ; i < st1.countTokens();){
	    if(!tokenMatching(st1.nextToken(),st2.nextToken())){
		// �g�[�N������ł��}�b�`���O�Ɏ��s�����玸�s
		return false;
	    }
	}
	
	// �Ō�܂� O.K. �Ȃ琬��
	return true;
    }

    boolean tokenMatching(String token1,String token2){
	//System.out.println(token1+"<->"+token2);
	if(token1.equals(token2)) return true;
	if( var(token1) && !var(token2)) return varMatching(token1,token2);
	if(!var(token1) &&  var(token2)) return varMatching(token2,token1);
	return false;
    }

    boolean varMatching(String vartoken,String token){
	if(vars.containsKey(vartoken)){
	    if(token.equals(vars.get(vartoken))){
		return true;
	    } else {
		return false;
	    }
	} else {
	    vars.put(vartoken,token);
	}
	return true;
    }

    boolean var(String str1){
	// �擪�� ? �Ȃ�ϐ�
	return str1.startsWith("?");
    }

}
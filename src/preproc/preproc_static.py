import polars as pl
from .utils import cast_to_date, cast_to_float, cast_to_int, cast_to_str 

def main_preproc_static(df_static):
    
    date_columns = return_date_columns()
    cat_columns = return_categorical_columns()
    num_columns = return_numerical_columns()
    df_static = cast_to_date(df_static, date_columns)
    df_static = cast_to_float(df_static, num_columns)
    df_static = cast_to_str(df_static, cat_columns)
    df_static = compute_date_distance_from_col(df_static, date_columns, "date_decision")
    return df_static


def compute_date_distance_from_col(df, date_col_list, date_col_to_compare):
    new_cols = []
    for col in date_col_list:
        df = df.with_columns((pl.col(date_col_to_compare) - pl.col(col)).dt.total_days().alias(f"{col}_days_from_appl"))
        new_cols.append(f"{col}_days_from_appl")
    return df, new_cols


def return_date_columns():
    date_columns = ['datefirstoffer_1144D',
        'datelastinstal40dpd_247D',
        'datelastunpaid_3546854D',
        'dtlastpmtallstes_4499206D',
        'firstclxcampaign_1125D',
        'firstdatedue_489D',
        'lastactivateddate_801D',
        'lastapplicationdate_877D',
        'lastapprdate_640D',
        'lastdelinqdate_224D',
        'lastrejectdate_50D',
        'lastrepayingdate_696D',
        'maxdpdinstldate_3546855D',
        'payvacationpostpone_4187118D',
        'validfrom_1069D']
    return date_columns

def return_categorical_columns():
    cat_columns = ['bankacctype_710L',
 'cardtype_51L',
 'credtype_322L',
 'disbursementtype_67L',
 'equalitydataagreement_891L',
 'equalityempfrom_62L',
 'inittransactioncode_186L',
 'isbidproduct_1095L',
 'isbidproductrequest_292L',
 'isdebitcard_729L',
 'lastapprcommoditycat_1041M',
 'lastapprcommoditytypec_5251766M',
 'lastcancelreason_561M',
 'lastrejectcommoditycat_161M',
 'lastrejectcommodtypec_5251769M',
 'lastrejectreason_759M',
 'lastrejectreasonclient_4145040M',
 'lastst_736L',
 'opencred_647L',
 'paytype1st_925L',
 'paytype_783L',
 'previouscontdistrict_112M',
 'twobodfilling_608L',
 'typesuite_864L']
    return cat_columns

def return_numerical_columns():
    num_columns = ['actualdpdtolerance_344P',
 'amtinstpaidbefduel24m_4187115A',
 'annuity_780A',
 'annuitynextmonth_57A',
 'applicationcnt_361L',
 'applications30d_658L',
 'applicationscnt_1086L',
 'applicationscnt_464L',
 'applicationscnt_629L',
 'applicationscnt_867L',
 'avgdbddpdlast24m_3658932P',
 'avgdbddpdlast3m_4187120P',
 'avgdbdtollast24m_4525197P',
 'avgdpdtolclosure24_3658938P',
 'avginstallast24m_3658937A',
 'avglnamtstart24m_4525187A',
 'avgmaxdpdlast9m_3716943P',
 'avgoutstandbalancel6m_4187114A',
 'avgpmtlast12m_4525200A',
 'clientscnt12m_3712952L',
 'clientscnt3m_3712950L',
 'clientscnt6m_3712949L',
 'clientscnt_100L',
 'clientscnt_1022L',
 'clientscnt_1071L',
 'clientscnt_1130L',
 'clientscnt_136L',
 'clientscnt_157L',
 'clientscnt_257L',
 'clientscnt_304L',
 'clientscnt_360L',
 'clientscnt_493L',
 'clientscnt_533L',
 'clientscnt_887L',
 'clientscnt_946L',
 'cntincpaycont9m_3716944L',
 'cntpmts24_3658933L',
 'commnoinclast6m_3546845L',
 'credamount_770A',
 'currdebt_22A',
 'currdebtcredtyperange_828A',
 'daysoverduetolerancedd_3976961L',
 'deferredmnthsnum_166L',
 'disbursedcredamount_1113A',
 'downpmt_116A',
 'eir_270L',
 'homephncnt_628L',
 'inittransactionamount_650A',
 'interestrate_311L',
 'interestrategrace_34L',
 'lastapprcredamount_781A',
 'lastdependentsnum_448L',
 'lastotherinc_902A',
 'lastotherlnsexpense_631A',
 'lastrejectcredamount_222A',
 'maininc_215A',
 'mastercontrelectronic_519L',
 'mastercontrexist_109L',
 'maxannuity_159A',
 'maxannuity_4075009A',
 'maxdbddpdlast1m_3658939P',
 'maxdbddpdtollast12m_3658940P',
 'maxdbddpdtollast6m_4187119P',
 'maxdebt4_972A',
 'maxdpdfrom6mto36m_3546853P',
 'maxdpdinstlnum_3546846P',
 'maxdpdlast12m_727P',
 'maxdpdlast24m_143P',
 'maxdpdlast3m_392P',
 'maxdpdlast6m_474P',
 'maxdpdlast9m_1059P',
 'maxdpdtolerance_374P',
 'maxinstallast24m_3658928A',
 'maxlnamtstart6m_4525199A',
 'maxoutstandbalancel12m_4187113A',
 'maxpmtlast3m_4525190A',
 'mindbddpdlast24m_3658935P',
 'mindbdtollast24m_4525191P',
 'mobilephncnt_593L',
 'monthsannuity_845L',
 'numactivecreds_622L',
 'numactivecredschannel_414L',
 'numactiverelcontr_750L',
 'numcontrs3months_479L',
 'numincomingpmts_3546848L',
 'numinstlallpaidearly3d_817L',
 'numinstls_657L',
 'numinstlsallpaid_934L',
 'numinstlswithdpd10_728L',
 'numinstlswithdpd5_4187116L',
 'numinstlswithoutdpd_562L',
 'numinstmatpaidtearly2d_4499204L',
 'numinstpaid_4499208L',
 'numinstpaidearly3d_3546850L',
 'numinstpaidearly3dest_4493216L',
 'numinstpaidearly5d_1087L',
 'numinstpaidearly5dest_4493211L',
 'numinstpaidearly5dobd_4499205L',
 'numinstpaidearly_338L',
 'numinstpaidearlyest_4493214L',
 'numinstpaidlastcontr_4325080L',
 'numinstpaidlate1d_3546852L',
 'numinstregularpaid_973L',
 'numinstregularpaidest_4493210L',
 'numinsttopaygr_769L',
 'numinsttopaygrest_4493213L',
 'numinstunpaidmax_3546851L',
 'numinstunpaidmaxest_4493212L',
 'numnotactivated_1143L',
 'numpmtchanneldd_318L',
 'numrejects9m_859L',
 'pctinstlsallpaidearl3d_427L',
 'pctinstlsallpaidlat10d_839L',
 'pctinstlsallpaidlate1d_3546856L',
 'pctinstlsallpaidlate4d_3546849L',
 'pctinstlsallpaidlate6d_3546844L',
 'pmtnum_254L',
 'posfpd10lastmonth_333P',
 'posfpd30lastmonth_3976960P',
 'posfstqpd30lastmonth_3976962P',
 'price_1097A',
 'sellerplacecnt_915L',
 'sellerplacescnt_216L',
 'sumoutstandtotal_3546847A',
 'sumoutstandtotalest_4493215A',
 'totaldebt_9A',
 'totalsettled_863A',
 'totinstallast1m_4525188A']
    return num_columns
    